#include "yolo.h"
#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>
#include<ctime>

using namespace std;
using namespace cv;
using namespace dnn;

static float save_two_decimal(float i)
{
	return round(i * 100) / 100;
}
static double timestamp_now_float() 
{
	return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

static string get_file_name(const string& path, bool include_suffix)
{

	if (path.empty())
		return "";
	// �ļ��� images/23.jpg ���� images\\23.jpg ���ǿ��ܵ�
	int p = path.rfind('/');  // �Ӻ���ǰ���ҵ�һ��б��   
	int e = path.rfind('\\');
	p = std::max(p, e);
	p += 1; // ��־λ��������һ��ƫ��������Ϊfile name��Ӧ����ʼ����

	//include suffix�� ����������׺�� jpg
	if (include_suffix)
		return path.substr(p); //�Ӵ�

	int u = path.rfind('.');
	if (u == -1) //���û���ҵ� �� �ַ�
		return path.substr(p);

	if (u <= p)
		u = path.size();
	return path.substr(p, u - p); //�����������׺jpg�� �����м���ַ���
}


int main()
{
	string model_path = "./weights/best.onnx"; //ģ��·��
	vector<cv::String> files_;  // typedef std::string cv::String����������һ������
	files_.reserve(10000);

	cv::glob("images/*.jpg", files_, false); // ͼƬ·��
	string save_root = "result"; // �������Ŀ¼

	vector<string> files(files_.begin(), files_.end()); // ���������vector��ģ�ʵ�������������죿
	for (auto c : files)
		cout << c << endl;
	//ʹ��images_vec�����洢���е�ͼƬMat
	vector<cv::Mat> images_vec; 
	for (int i = 0; i < files.size(); ++i) 
	{
		auto image = cv::imread(files[i]);
		images_vec.emplace_back(image);
	}
	//��ȡ�������ɫ����
	vector<Scalar> color_vec; 
	for (int i = 0; i < 20; i++)
	{
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color_vec.push_back(Scalar(b, g, r));
	}
	// �����ʼ��
	auto begin_timer = timestamp_now_float();

	Yolo test;
	Net net;
	// ���硢ģ��·�����Ƿ�ʹ��cuda
	if (test.Init(net, model_path, true))
		cout << "read net ok!" << endl;
	else
	{
		cout << "ģ�ͳ�ʼ������" << endl;
		return -1;
	}
	vector<Output> out_result; //ʹ�øñ����洢���еĿ�Output���Զ���Ľṹ��
	for (int i = 0; i < images_vec.size(); i++)
	{
		//auto img = images_vec[i];
		bool detec_flag = test.Detect(images_vec[i], net, out_result);
		if (detec_flag)
		{
			test.drawPred(images_vec[i], out_result, color_vec);
			string file_name = get_file_name(files[i], true);
			string save_path = cv::format("%s/%s", save_root.c_str(), file_name.c_str()); // ����Ҫ��stringת��c���ַ���
			cv::imwrite(save_path, images_vec[i]);
			out_result.clear(); //��ձ������������
		}
	}
	float inference_average_time = (timestamp_now_float() - begin_timer) / images_vec.size();
	cout << endl;
	cout << "average cost time : " << save_two_decimal(inference_average_time) << "ms" << endl;
	system("pause");
	return 0;
}