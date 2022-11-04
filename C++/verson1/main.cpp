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
	// 文件名 images/23.jpg 或者 images\\23.jpg 都是可能的
	int p = path.rfind('/');  // 从后往前，找第一个斜杠   
	int e = path.rfind('\\');
	p = std::max(p, e);
	p += 1; // 标志位往后，增加一个偏移量，即为file name对应的起始下面

	//include suffix， 即，包含后缀， jpg
	if (include_suffix)
		return path.substr(p); //子串

	int u = path.rfind('.');
	if (u == -1) //如果没有找到 点 字符
		return path.substr(p);

	if (u <= p)
		u = path.size();
	return path.substr(p, u - p); //如果不包含后缀jpg， 返回中间的字符串
}


int main()
{
	string model_path = "./weights/best.onnx"; //模型路径
	vector<cv::String> files_;  // typedef std::string cv::String，这两个是一个东西
	files_.reserve(10000);

	cv::glob("images/*.jpg", files_, false); // 图片路径
	string save_root = "result"; // 结果保存目录

	vector<string> files(files_.begin(), files_.end()); // 这个类似于vector类的，实例化，拷贝构造？
	for (auto c : files)
		cout << c << endl;
	//使用images_vec变量存储所有的图片Mat
	vector<cv::Mat> images_vec; 
	for (int i = 0; i < files.size(); ++i) 
	{
		auto image = cv::imread(files[i]);
		images_vec.emplace_back(image);
	}
	//获取随机的颜色参数
	vector<Scalar> color_vec; 
	for (int i = 0; i < 20; i++)
	{
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color_vec.push_back(Scalar(b, g, r));
	}
	// 网络初始化
	auto begin_timer = timestamp_now_float();

	Yolo test;
	Net net;
	// 网络、模型路径、是否使用cuda
	if (test.Init(net, model_path, true))
		cout << "read net ok!" << endl;
	else
	{
		cout << "模型初始化错误" << endl;
		return -1;
	}
	vector<Output> out_result; //使用该变量存储所有的框，Output是自定义的结构体
	for (int i = 0; i < images_vec.size(); i++)
	{
		//auto img = images_vec[i];
		bool detec_flag = test.Detect(images_vec[i], net, out_result);
		if (detec_flag)
		{
			test.drawPred(images_vec[i], out_result, color_vec);
			string file_name = get_file_name(files[i], true);
			string save_path = cv::format("%s/%s", save_root.c_str(), file_name.c_str()); // 这里要把string转成c的字符串
			cv::imwrite(save_path, images_vec[i]);
			out_result.clear(); //清空变量里面的内容
		}
	}
	float inference_average_time = (timestamp_now_float() - begin_timer) / images_vec.size();
	cout << endl;
	cout << "average cost time : " << save_two_decimal(inference_average_time) << "ms" << endl;
	system("pause");
	return 0;
}