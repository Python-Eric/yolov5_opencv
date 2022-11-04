#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


struct Output 
{
	int id;             //������id
	float confidence;   //������Ŷ�
	cv::Rect box;       //���ο�
};

class Yolo 
{
public:
	Yolo() 
	{
		cout << "yolo�ഴ��" << endl;
	}
	
	bool Init(cv::dnn::Net& net, std::string& netPath, bool isCuda);
	cv::Mat LetterBox(Mat& src); //letterbox
	bool Detect(cv::Mat& SrcImg, cv::dnn::Net& net, std::vector<Output>& output);
	// ������õ��Ŀ� ӳ���ԭͼ
	cv::Rect dst2src(Rect& det_rect);
	void drawPred(cv::Mat& img, std::vector<Output> result, std::vector<cv::Scalar> color);
	~Yolo() 
	{
		cout << "Yolo������" << endl;
	}

private:
	bool isLetter_flag = true; // һ�㶼Ҫ��letterbox
	const float netStride[3] = { 8, 16, 32 };
	const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
	int netWidth = 640;
	int netHeight = 640;// ��������Ϊ640�������޸ĵ�

	float boxThreshold = 0.35;
	float classThreshold = 0.25;

	float nmsThreshold = 0.45;
	// ratio��topPad�ȣ�ֻ�������ó�ʼֵ����Ϊ�˺����޸ģ��Լ����dst2src
	float ratio = 1.0;//���ŵı�����������Ϊ1.0�� yolo.cpp�����޸�
	int topPad = 0, btmPad = 0, leftPad = 0, rightPad = 0;
	// std::vector<std::string> className = { "person", "hat" };

	std::vector<std::string> className = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
         "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
};