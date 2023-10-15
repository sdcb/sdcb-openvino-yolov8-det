using OpenCvSharp.Dnn;
using OpenCvSharp;
using Sdcb.OpenVINO.Natives;
using Sdcb.OpenVINO;
using System.Diagnostics;
using System.Xml.Linq;
using System.Xml.XPath;

public class Program
{
    static unsafe void Main()
    {
        string modelFile = @".\yolov8n_openvino_model\yolov8n.xml";
        string[] dicts = XDocument.Load(modelFile)
            .XPathSelectElement(@"/net/rt_info/model_info/labels")!.Attribute("value")!.Value
            .Split(' ');
        using VideoCapture vc = new(1);

        using Model rawModel = OVCore.Shared.ReadModel(modelFile);
        using PrePostProcessor pp = rawModel.CreatePrePostProcessor();
        using (PreProcessInputInfo inputInfo = pp.Inputs.Primary)
        {
            inputInfo.TensorInfo.Layout = Layout.NHWC;
            inputInfo.ModelInfo.Layout = Layout.NCHW;
        }
        using Model m = pp.BuildModel();
        using CompiledModel cm = OVCore.Shared.CompileModel(m, "CPU");
        using InferRequest ir = cm.CreateInferRequest();

        Shape inputShape = m.Inputs.Primary.Shape;
        Size2f sizeRatio = new Size2f(1f * vc.FrameWidth / inputShape[2], 1f * vc.FrameHeight / inputShape[1]);
        while (vc.Grab())
        {
            using Mat src = vc.RetrieveMat();
            Stopwatch stopwatch = new();
            using Mat resized = src.Resize(new Size(inputShape[2], inputShape[1]));
            using Mat f32 = new();
            resized.ConvertTo(f32, MatType.CV_32FC3, 1.0 / 255);

            using (Tensor input = Tensor.FromRaw(
                new ReadOnlySpan<byte>((void*)f32.Data, (int)((nint)f32.DataEnd - f32.DataStart)),
                new Shape(1, f32.Rows, f32.Cols, 3),
                ov_element_type_e.F32))
            {
                ir.Inputs.Primary = input;
            }

            double preprocessTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Restart();

            ir.Run();
            double inferTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Restart();

            using (Tensor output = ir.Outputs.Primary)
            {
                ReadOnlySpan<float> data = output.GetData<float>();
                DetectionResult[] results = DetectionResult.FromYolov8DetectionResult(data, output.Shape, sizeRatio, dicts);
                double postprocessTime = stopwatch.Elapsed.TotalMilliseconds;
                stopwatch.Stop();
                double totalTime = preprocessTime + inferTime + postprocessTime;

                Cv2.PutText(src, $"Preprocess: {preprocessTime:F2}ms", new Point(10, 20), HersheyFonts.HersheyPlain, 1, Scalar.Red);
                Cv2.PutText(src, $"Infer: {inferTime:F2}ms", new Point(10, 40), HersheyFonts.HersheyPlain, 1, Scalar.Red);
                Cv2.PutText(src, $"Postprocess: {postprocessTime:F2}ms", new Point(10, 60), HersheyFonts.HersheyPlain, 1, Scalar.Red);
                Cv2.PutText(src, $"Total: {totalTime:F2}ms", new Point(10, 80), HersheyFonts.HersheyPlain, 1, Scalar.Red);

                foreach (DetectionResult r in results)
                {
                    Cv2.PutText(src, $"{r.Class}:{r.Confidence:P0}", r.Rect.TopLeft, HersheyFonts.HersheyPlain, 1, Scalar.Blue);
                    Cv2.Rectangle(src, r.Rect, Scalar.Blue, thickness: 2);
                }

                Cv2.ImShow("frame", src);
                Cv2.WaitKey(1);
            }
        }
    }
}

public record DetectionResult(int ClassId, string Class, Rect Rect, float Confidence)
{
    public static DetectionResult[] FromYolov8DetectionResult(ReadOnlySpan<float> tensorData, Shape shape, Size2f sizeRatio, string[] dicts)
    {
        // tensorData: 1x84x8400=705600xF32
        // shape: 1x84x8400, 84=(x, y, width, height)+80 class confidences, 8400=possible object count(code should for loop 8400 first)
        float[] t = Transpose(tensorData, shape[1], shape[2]);
        List<DetectionResult> detResults = new();

        int objectCount = shape[2];
        int clsRowCount = shape[1];
        if (dicts.Length != clsRowCount - 4) throw new ArgumentException($"dicts length {dicts.Length} does not match shape cls row count{clsRowCount}.");
        for (int i = 0; i < objectCount; i++)
        {
            ReadOnlySpan<float> rectData = t[(i * clsRowCount)..(i * clsRowCount + 4)];
            ReadOnlySpan<float> confidenceInfo = t[(i * clsRowCount + 4)..(i * clsRowCount + clsRowCount)];
            int maxConfidenceClsId = IndexOfMax(confidenceInfo);
            float confidence = confidenceInfo[maxConfidenceClsId];

            int centerX = (int)(rectData[0] * sizeRatio.Width);
            int centerY = (int)(rectData[1] * sizeRatio.Height);
            int width = (int)(rectData[2] * sizeRatio.Width);
            int height = (int)(rectData[3] * sizeRatio.Height);
            detResults.Add(new DetectionResult(
                maxConfidenceClsId, dicts[maxConfidenceClsId],
                new Rect(centerX - width / 2, centerY - height / 2, width, height),
                confidence));
        }

        CvDnn.NMSBoxes(detResults.Select(x => x.Rect), detResults.Select(x => x.Confidence), scoreThreshold: 0.5f, nmsThreshold: 0.5f, out int[] indices);
        return detResults.Where((x, i) => indices.Contains(i)).ToArray();
    }

    static int IndexOfMax(ReadOnlySpan<float> data)
    {
        if (data.Length == 0) throw new ArgumentException("The provided data span is null or empty.");

        // 初始化最大值及其索引
        int maxIndex = 0;
        float maxValue = data[0];

        // 遍历跨度查找最大值及其索引
        for (int i = 1; i < data.Length; i++)
        {
            if (data[i] > maxValue)
            {
                maxValue = data[i];
                maxIndex = i;
            }
        }

        // 返回最大值及其索引
        return maxIndex;
    }

    static unsafe float[] Transpose(ReadOnlySpan<float> tensorData, int rows, int cols)
    {
        float[] transposedTensorData = new float[tensorData.Length];

        fixed (float* pTensorData = tensorData)
        {
            fixed (float* pTransposedData = transposedTensorData)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        // Index in the original tensor
                        int index = i * cols + j;

                        // Index in the transposed tensor
                        int transposedIndex = j * rows + i;

                        pTransposedData[transposedIndex] = pTensorData[index];
                    }
                }
            }
        }

        return transposedTensorData;
    }
}