#include <iostream>
#include <algorithm>
#include "tensorRTplugin/tensorNet.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "util/cuda/cudaRGB.h"
#include "util/loadImage.h"
#include <chrono>

using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace cv;

const char* model  = "/home/student/Downloads/ssd_deploy_iplugin.prototxt";
const char* weight = "/home/student/Downloads/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
const char* label  = "/home/student/Projects/caffe-ssd/data/VOC0712/labelmap_voc.prototxt";

static const uint32_t BATCH_SIZE = 1;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT1 = "mbox_conf_softmax";
const char* OUTPUT2 = "mbox_loc";
const char* OUTPUT3 = "mbox_priorbox";
const char* OUTPUT_BLOB_NAME = "detection_out";


class Timer {
 public:
  void tic() {
    start_ticking_ = true;
    start_ = std::chrono::high_resolution_clock::now();
  }
  void toc() {
    if (!start_ticking_)return;
    end_ = std::chrono::high_resolution_clock::now();
    start_ticking_ = false;
    double t = std::chrono::duration<double, std::milli>(end_ - start_).count();
    std::cout << "Time: " << t << " ms" << std::endl;
  }
 private:
  bool start_ticking_ = false;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

void CheckImageSize(cv::Mat* image, std::size_t size) {
  if (image->rows != size || image->cols != size)
    cv::resize(*image, *image, cv::Size(size, size));
}

/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}

cudaError_t cudaPreImageNetMean( float3* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value);

int main()
{
    VideoCapture cap("/home/student/Videos/parking_slot_video/surround_view/20161109-03.avi");

    if(!cap.isOpened())
    {
        cout<<"There is no video in this location"<<endl;
        return -1;
    }

    TensorNet tensorNet;
    tensorNet.caffeToTRTModel( model, weight, std::vector<std::string>{ OUTPUT_BLOB_NAME }, BATCH_SIZE);
    tensorNet.createInference();

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);
    DimsCHW dimsOut1  = tensorNet.getTensorDims(OUTPUT1);
    DimsCHW dimsOut2  = tensorNet.getTensorDims(OUTPUT2);
    DimsCHW dimsOut3  = tensorNet.getTensorDims(OUTPUT3);

    cout << "INPUT Tensor Shape is: C: "  <<dimsData.c()<< "  H: "<<dimsData.h()<<"  W:  "<<dimsData.w()<<endl;
    cout << "mbox_conf_softmax Tensor Shape is: C: "<<dimsOut1.c()<<"  H: "<<dimsOut1.h()<<"  W: "<<dimsOut1.w()<<endl;
    cout << "mbox_loc Tensor Shape is: C: "<<dimsOut2.c()<<"  H: "<<dimsOut2.h()<<"  W: "<<dimsOut2.w()<<endl;
    cout << "mbox_priorbox Tensor Shape is: C: "<<dimsOut3.c()<<"  H: "<<dimsOut3.h()<<"  W: "<<dimsOut3.w()<<endl;
    cout << "OUTPUT Tensor Shape is: C: "<<dimsOut.c()<<"  H: "<<dimsOut.h()<<"  W: "<<dimsOut.w()<<endl;

    float* data    = allocateMemory( dimsData , (char*)"input blob");
    // float* output1  = allocateMemory( dimsOut1  , (char*)"output blob");
    // float* output2  = allocateMemory( dimsOut2  , (char*)"output blob");
    // float* output3  = allocateMemory( dimsOut3  , (char*)"output blob");
    float* output  = allocateMemory( dimsOut  , (char*)"output blob");

    int height = 300;
    int width  = 300;

    Mat frame;
    Mat frame_float;

    /* *
     * @TODO: Replace imgCPU -> h_img ||| imgGPu -> d_img
     * */

    void* imgCPU;
    void* imgCUDA;
    Timer timer;

    while (true)
    {
        std::string image_index;
        std::cout << "Enter image path:";
        std::cin >> image_index;
        frame = cv::imread("/home/student/data/VOCdevkit/VOC2007/JPEGImages/00000" + image_index + ".jpg", IMREAD_COLOR);
        resize(frame, frame, Size(300,300));
        const size_t size = width * height * sizeof(float3);

        if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
        {
            cout <<"Cuda Memory allocation error occured."<<endl;
            return false;
        }
        if( !loadImageBGR( frame , (float3**)&imgCPU, (float3**)&imgCUDA, &height, &width))
        {
            printf("failed to load image '%s'\n", "Image");
            return 0;
        }

        if( CUDA_FAILED(cudaPreImageNetMean( (float3*)imgCUDA, width, height, data, dimsData.w(), dimsData.h(), make_float3(123.0f,117.0f,104.0f))))
        {
            cout <<"Cuda pre image net mean failed. " <<endl;
            return 0;
        }

        void* buffers[] = { imgCUDA, output };

        timer.tic();
        tensorNet.imageInference( buffers, 2, BATCH_SIZE);
        timer.toc();

        for (int k=0; k<10; k++)
        {
            std::cout << output[7*k+0] << " "
                      << output[7*k+1] << " "
                      << output[7*k+2] << " "
                      << output[7*k+3] << " "
                      << output[7*k+4] << " "
                      << output[7*k+5] << " "
                      << output[7*k+6] << "\n";
            if(output[7*k+1] == -1) continue;
            float xmin = 300 * output[7*k + 3];
            float ymin = 300 * output[7*k + 4];
            float xmax = 300 * output[7*k + 5];
            float ymax = 300 * output[7*k + 6];
            using cv::Point2f;
            using cv::line;
            using cv::Scalar;
            Point2f a = Point2f(xmin, ymin);
            Point2f b = Point2f(xmin, ymax);
            Point2f c = Point2f(xmax, ymax);
            Point2f d = Point2f(xmax, ymin);
            line(frame, a, b, Scalar(0.0, 255.0, 255.0));
            line(frame, b, c, Scalar(0.0, 255.0, 255.0));
            line(frame, c, d, Scalar(0.0, 255.0, 255.0));
            line(frame, d, a, Scalar(0.0, 255.0, 255.0));
            //std::cout << xmin << ", " << ymin << ", " << xmax << ", " << ymax << "\n";        }
        }
        imshow("Objects Detected", frame);
        waitKey(1);
    }

    CUDA(cudaFreeHost(imgCPU));
    tensorNet.destroy();

    return 0;

}
