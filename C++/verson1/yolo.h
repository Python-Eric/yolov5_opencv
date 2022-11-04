#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


struct Output 
{
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
};

class Yolo 
{
public:
	Yolo() 
	{
		cout << "yolo类创建" << endl;
	}
	
	bool Init(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	cv::Mat LetterBox(Mat& src); //letterbox
	bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output);
	// 将输出得到的框 映射回原图
	cv::Rect dst2src(Rect& det_rect);
	void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);
	~Yolo() 
	{
		cout << "Yolo类析构" << endl;
	}

private:
	bool isLetter_flag = true; // 一般都要用letterbox
	const float netStride[3] = { 8, 16, 32 };
	const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
	int netWidth = 640;
	int netHeight = 640;// 网络输入为640。可以修改的

	float boxThreshold = 0.35;
	float classThreshold = 0.25;

	float nmsThreshold = 0.45;
	// ratio和topPad等，只所以设置初始值，是为了后续修改，以及最后dst2src
	float ratio = 1.0;//缩放的比例，先设置为1.0， yolo.cpp可以修改
	int topPad = 0, btmPad = 0, leftPad = 0, rightPad = 0;
	// std::vector<std::string> className = { "person", "hat" };

	std::vector<std::string> className = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
         "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
};