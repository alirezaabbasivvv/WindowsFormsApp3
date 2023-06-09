using DlibDotNet;
using Emgu.CV.CvEnum;
using Emgu.CV.Ocl;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.Face;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp3
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        public string BasePath = @"/Faces";
        private VideoCapture capture;
        public System.Windows.Forms.Timer GetFrame;
        public CascadeClassifier faceCascade;
        public Emgu.CV.CascadeClassifier faceCascade2;
        public ShapePredictor shapePredictor;
        List<string> Names = new List<string>();
        List<string> names;
        public Emgu.CV.Face.EigenFaceRecognizer recognizer;
        List<Dictionary<string, Mat>> imageList = new List<Dictionary<string, Mat>>();
        private Emgu.CV.Util.VectorOfInt labelList = new Emgu.CV.Util.VectorOfInt();
        private Emgu.CV.Util.VectorOfMat imageList2t = new Emgu.CV.Util.VectorOfMat();

        private void Form1_Load(object sender, EventArgs e)
        {
            faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
            shapePredictor = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat");
            LoadTraineImage();

            capture = new VideoCapture(0);
            GetFrame = new System.Windows.Forms.Timer();
            GetFrame.Interval = 1000;
            GetFrame.Tick += GetFrame_Tick;
            recognizer = new Emgu.CV.Face.EigenFaceRecognizer(imageList.Count);

            names = new List<string>();
            List<Emgu.CV.Mat> face1 = new List<Emgu.CV.Mat>();
            faceCascade2 = new Emgu.CV.CascadeClassifier("haarcascade_frontalface_default.xml");
            foreach (var faces in imageList)
            {
                for (int i = 0; i < faces.Count-1; i++)
                {

                    labelList.Push(new[] { i++ });
                    names.Add(faces.ElementAt(i).Key);
                    imageList2t.Push(mat2mat(faces.ElementAt(i).Value));
                    face1.Add(mat2mat(faces.ElementAt(i).Value));

                }

            }
            recognizer.Train(imageList2t, labelList);


        }
        private string FaceRecognition(Emgu.CV.Mat detectedface)
        {
            try
            {
                if (imageList.Count != 0)
                {
                    //Eigen Face Algorithm
                    Emgu.CV.Face.FaceRecognizer.PredictionResult result = recognizer.Predict(detectedface);
                    return names[result.Label];

                }
                else
                {
                    return string.Empty;
                }
            }catch(Exception ex)
            {
                return string.Empty;
            }
        }
        private void GetFrame_Tick(object sender, EventArgs e)
        {
            using (var frame = new Mat())
            {
                capture.Read(frame);
                var grayImage = new Mat();
                Cv2.CvtColor(frame, grayImage, ColorConversionCodes.BGR2GRAY);
                Rect[] faces = faceCascade.DetectMultiScale(grayImage, scaleFactor: 1.1, minNeighbors: 3, flags: 0, minSize: new OpenCvSharp.Size(30, 30));
                foreach (Rect face in faces)
                {
                    Cv2.Rectangle(frame, face, Scalar.Aqua, thickness: 2);
                    DlibDotNet.Rectangle dlibRect = new DlibDotNet.Rectangle(face.X, face.Y, face.X + face.Width, face.Y + face.Height);
                    FullObjectDetection landmarks = shapePredictor.Detect(ConvertMatToArray2D(grayImage), dlibRect);
                    DlibDotNet.Point nose = landmarks.GetPart(30);
                    Cv2.Circle(frame, (int)nose.X, (int)nose.Y, radius: 10, Scalar.Red, thickness: 2);
                    Console.WriteLine("x={0} , y ={1}", nose.X, nose.Y);
                    var detec = FaceRecognition(mat2mat(grayImage));
                    if(!string.IsNullOrEmpty(detec))
                    {
                        label2.Text=detec;
                    }
                    else
                    {
                        label2.Text="";

                    }
                }
                pictureBox1.Image = BitmapConverter.ToBitmap(frame);
            }
        }

        //public double comparefaces(OpenCvSharp.Mat face1, OpenCvSharp.Mat face2)
        //{

        //    Emgu.CV.Mat image1 = Emgu.CV.CvInvoke.Imread("path/to/image1.jpg", ImreadModes.Grayscale);
        //    Emgu.CV.Mat image2 = Emgu.CV.CvInvoke.Imread("path/to/image2.jpg", (Emgu.CV.CvEnum.ImreadModes)ImreadModes.Grayscale);
        //    Emgu.CV.CvInvoke.GaussianBlur(image1, image1, new System.Drawing.Size(3, 3), 0);
        //    Emgu.CV.CvInvoke.GaussianBlur(image2, image2, new System.Drawing.Size(3, 3), 0);

        //    //Detect facial landmarks
        //    //TODO: Implement facial landmark detection

        //    //Extract facial features using LBPH algorithm
        //    LBPHFaceRecognizer lbph = new LBPHFaceRecognizer();
        //    lbph.Train(new[] { image1, image2 }, new[] { 1, 2 });
        //    var features1 = lbph.Predict(image1).Histogram;
        //    var features2 = lbph.Predict(image2).Histogram;

        //    //Compare the extracted features using cosine similarity
        //    double similarity = Emgu.CV.CvInvoke.Compare(features1, features2, Emgu.CV.CvEnum.HistogramCompMethod.Intersect);
        //    return similarity;

        //}
        public Emgu.CV.Mat mat2mat(OpenCvSharp.Mat input)
        {

            var emguMat = new Emgu.CV.Mat();
            Emgu.CV.CvInvoke.Imdecode(input.ToBytes(), Emgu.CV.CvEnum.ImreadModes.Grayscale, emguMat);
            return emguMat;

        }
        public void LoadTraineImage()
        {
            DirectoryInfo directory = new DirectoryInfo(BasePath);
            if (directory.Exists)
            {
                foreach (var dir in directory.GetDirectories())
                {
                    string label = dir.Name;
                    string subpath = BasePath + "\\" + label;
                    var dicti = new Dictionary<string, Mat>();
                    DirectoryInfo subdirectory = new DirectoryInfo(subpath);
                    var items = subdirectory.GetFiles("*.jpg");
                    if (items.Length > 0)
                    {
                        foreach (var item in subdirectory.GetFiles("*.jpg"))
                        {
                            var img = Cv2.ImRead(Path.Combine(subpath, item.Name));
                            dicti.Add(item.Name.Split('.')[0], img);
                        }
                        Names.Add(dir.Name);
                        imageList.Add(dicti);
                    }
                }
            }
            else
            {
                MessageBox.Show("not image for trian");
            }
        }

        public enum TestFace
        {
            //+
            Left = 1,
            //-
            Right = 2,
        }
        public void Add_Face(Mat frontFace, DlibDotNet.Point point)
        {
            DirectoryInfo directory = new DirectoryInfo(BasePath);
            if (directory.Exists)
            {
                if (!Directory.Exists(BasePath + txt_newFaceName.Text))
                {
                    string path = BasePath + txt_newFaceName.Text;
                    directory.CreateSubdirectory(txt_newFaceName.Text);
                    var bmp = BitmapConverter.ToBitmap(frontFace);
                    var pa = Path.Combine(BasePath + "\\" + txt_newFaceName.Text, "0.jpg");
                    bmp.Save(pa, System.Drawing.Imaging.ImageFormat.Jpeg);
                    label1.Text = "please left";
                    Thread.Sleep(3000);
                    Routeathead(TestFace.Left, point);
                    label1.Text = "please right";
                    Thread.Sleep(3000);
                    Routeathead(TestFace.Right, point);

                }

                else
                    MessageBox.Show("this name alerdy exists", "", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                directory.Create();
                MessageBox.Show("try againe", "", MessageBoxButtons.OK, MessageBoxIcon.Error);

            }
        }
        public void Routeathead(TestFace rout, DlibDotNet.Point poing)
        {
            Dictionary<int, Mat> list = new Dictionary<int, Mat>();
            int index = 0;
            int rate = poing.X;

            GetFrame.Stop();
            for (int i = 0; i < 5; i = i)
            {
                Thread.Sleep(1000);
                Mat img = new Mat();
                var grayImage = new Mat();
                capture.Read(img);
                Cv2.CvtColor(img, grayImage, ColorConversionCodes.BGR2GRAY);
                Rect[] faces = faceCascade.DetectMultiScale(grayImage, scaleFactor: 1.1, minNeighbors: 3, flags: 0, minSize: new OpenCvSharp.Size(30, 30));
                pictureBox1.Image = BitmapConverter.ToBitmap(img);
                if (faces.Length > 0)
                {
                    DlibDotNet.Rectangle dlibRect = new DlibDotNet.Rectangle(faces[0].X, faces[0].Y, faces[0].X + faces[0].Width, faces[0].Y + faces[0].Height);
                    FullObjectDetection landmarks = shapePredictor.Detect(ConvertMatToArray2D(grayImage), dlibRect);
                    DlibDotNet.Point nose = landmarks.GetPart(30);

                    switch (rout)
                    {
                        case TestFace.Left:
                            if (nose.X > rate + 3 && nose.X < rate + 15)
                            {
                                index++;
                                rate += nose.X - rate;
                                list.Add(index, grayImage);
                                i++;
                                MessageBox.Show("ok litel to left more");
                            }
                            else
                            {
                                MessageBox.Show("try againe left");
                            }
                            break;
                        case TestFace.Right:
                            if (nose.X < rate - 5 && nose.X > rate - 15)
                            {
                                index--;
                                rate -= rate - nose.X;
                                list.Add(index, grayImage);
                                i++;
                                MessageBox.Show("ok litel to right more");

                            }
                            {

                                MessageBox.Show("try againe right");
                            }
                            break;
                    }
                    Thread.Sleep(100);
                }


            }
            foreach (var item in list)
            {
                try
                {
                    //Cv2.ImWrite(Path.Combine(BasePath + txt_newFaceName.Text, item.Key.ToString() + ".jpg"), item.Value);
                    var bmp = BitmapConverter.ToBitmap(item.Value);
                    bmp.Save(Path.Combine(BasePath + "\\" + txt_newFaceName.Text, item.Key.ToString() + ".jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                }
                catch (Exception ex) { }
            }

        }
        public void Procces_Face()
        {
        }
        public Array2D<byte> ConvertMatToArray2D(Mat mat)
        {
            // Create a new Array2D
            var array2D = new Array2D<byte>(mat.Height, mat.Width);

            // Convert Mat to byte array
            byte[] data = new byte[mat.Width * mat.Height];
            mat.GetArray(out data);

            // Copy the data to Array2D
            Marshal.Copy(data, 0, array2D.Data, data.Length);

            return array2D;
        }

        private void btn_start_Click(object sender, EventArgs e)
        {
            GetFrame.Start();
        }

        private void btn_add_Click(object sender, EventArgs e)
        {
            if (!string.IsNullOrEmpty(txt_newFaceName.Text))
            {
                using (var frame = new Mat())
                {
                    capture.Read(frame);
                    var grayImage = new Mat();
                    Cv2.CvtColor(frame, grayImage, ColorConversionCodes.BGR2GRAY);
                    Rect face = faceCascade.DetectMultiScale(grayImage, scaleFactor: 1.1, minNeighbors: 3, flags: 0, minSize: new OpenCvSharp.Size(30, 30))[0];
                    if (face != null)
                    {
                        DlibDotNet.Rectangle dlibRect = new DlibDotNet.Rectangle(face.X, face.Y, face.X + face.Width, face.Y + face.Height);
                        FullObjectDetection landmarks = shapePredictor.Detect(ConvertMatToArray2D(grayImage), dlibRect);
                        DlibDotNet.Point nose = landmarks.GetPart(30);
                        Add_Face(grayImage, nose);
                    }

                }
            }
        }
    }
}
