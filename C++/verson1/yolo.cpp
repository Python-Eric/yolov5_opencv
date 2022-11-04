#include"yolo.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;


static float save_two_decimal(float i)
{
	return round(i * 100) / 100;
}


static double timestamp_now_float()
{
	return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

bool Yolo::Init(Net& net, string& netPath, bool isCuda) 
{
	try 
	{
		net = readNetFromONNX(netPath);
	}
	catch (const std::exception&) 
	{
		return false;
	}
	//cuda
	if (isCuda) 
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	}
	//cpu
	else 
	{
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	
	return true;
}
Mat Yolo::LetterBox(Mat& src)
{
	if (src.empty())
	{
		cout << "输入图片路径有问题" << endl;
		return cv::Mat();
	}
	// 以hat.jpg为例， 宽650， 高315
	int in_w = src.cols;
	int in_h = src.rows;
	// 下面取float很关键，否则结果不对
	// 长边缩放， 因此，ratio为0.984
	ratio = min(float(netHeight) / in_h, float(netWidth) / in_w);
	int inside_w = round(in_w * ratio);
	int inside_h = round(in_h * ratio);

	float pad_w = netWidth - inside_w;
	float pad_h = netHeight - inside_h;

	Mat resize_img, blob;
	cv::resize(src, resize_img, cv::Size(inside_w, inside_h));
	cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
	
	pad_w = pad_w / 2.0;
	pad_h = pad_h / 2.0;

	//外层填充为灰色
	// pad_h代表高度方向的，自然是top和btm
	// pad_w 代表宽度方向的， 自然是left和right
	topPad = int(std::round(pad_h - 0.1));
	btmPad = int(std::round(pad_h + 0.1));
	leftPad = int(std::round(pad_w - 0.1));
	rightPad = int(std::round(pad_w + 0.1));

	cv::copyMakeBorder(resize_img, resize_img, topPad, btmPad, leftPad, rightPad, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	cv::dnn::blobFromImage(resize_img, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), false, false);
	return blob;
}
bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& result) 
{
	auto begin_timer_1 = timestamp_now_float();
	Mat input_blob = LetterBox(SrcImg);
	auto begin_timer_2 = timestamp_now_float();
	float pre_time = save_two_decimal((begin_timer_2 - begin_timer_1)) ; //  / 100
	net.setInput(input_blob); // 此处blob是四维， bchw
	std::vector<cv::Mat> output_blob;
	//vector<string> outputLayerName{"345","403", "461","output" };
	//net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
	net.forward(output_blob, net.getUnconnectedOutLayersNames());
	auto begin_timer_3 = timestamp_now_float();
	float infer_time = save_two_decimal((begin_timer_3 - begin_timer_2)) ; // / 100
	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes_output;//每个id矩形框
	
	int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
	
	float* pdata = (float*)output_blob[0].data;
	// n 表示特征层， 依次遍历 stride=8、16、32的检测头
	// 依次计算小目标、中目标、大目标
	for (int n = 0; n < 3; n++)
	{    //grid_x对应输出特征图的宽
		int grid_x = (int)(netWidth / netStride[n]);
		int grid_y = (int)(netHeight / netStride[n]);
		// 每个特征图有三个对应的anchor
		for (int q = 0; q < 3; q++) 
		{	//anchors
			const float anchor_w = netAnchors[n][q * 2];
			const float anchor_h = netAnchors[n][q * 2 + 1];
			for (int i = 0; i < grid_y; i++) 
			{
				for (int j = 0; j < grid_x; j++) 
				{
					float box_score = pdata[4]; ;//获取每一行的box框中含有某个物体的概率
					if (box_score >= boxThreshold) 
					{
						cv::Mat scores(1, className.size(), CV_32FC1, pdata + 5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = box_score * max_class_socre;
						if (max_class_socre >= classThreshold) 
						{
							float x = (pdata[0] * 2.f - 0.5f + j) * netStride[n];  //x
							float y = (pdata[1] * 2.f - 0.5f + i) * netStride[n];  //y
							float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;  //w
							float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  //h

							int x1 = (x - 0.5 * w);
							int y1 = (y - 0.5 * h);
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre);
							boxes_output.push_back(Rect(x1, y1, int(w), int(h)));
						}
					}
					pdata += net_width;//下一行
				}
			}
		}
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	dnn::NMSBoxes(boxes_output, confidences, classThreshold,nmsThreshold, nms_result);
	if (nms_result.size())
	{
		Output tmp;
		for (int i = 0; i < nms_result.size(); i++)
		{
			int idx = nms_result[i];
			Output tmp;
			tmp.id = classIds[idx];
			tmp.confidence = confidences[idx];
			tmp.box = boxes_output[idx];
			// 矩形框映射回原图上
			tmp.box = dst2src(tmp.box);
			result.push_back(tmp);
		}
	}
	auto begin_timer_4 = timestamp_now_float();
	float post_time = save_two_decimal((begin_timer_4 - begin_timer_3)) ; // / 100

	cout << "pre : " << pre_time << " ms, " << "infer : " << infer_time << " ms, " << "post :" << post_time << " ms " << endl;
	if (result.size())
		return true;
	else
		return false;
}
cv::Rect Yolo::dst2src(Rect& det_rect)
{
	int inside_x = det_rect.x - leftPad;
	int inside_y = det_rect.y - topPad;
	int ox = int(float(inside_x) / ratio);
	int oy = int(float(inside_y) / ratio);
	int ow = int(float(det_rect.width) / ratio);
	int oh = int(float(det_rect.height) / ratio);
	return cv::Rect(ox, oy, ow, oh);
}
void Yolo::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) 
{
	for (int i = 0; i < result.size(); i++)
	{
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		/*float conf = save_two_decimal(result[i].confidence);*/
		string conf = to_string(result[i].confidence).substr(0, 5); //
		string label = className[result[i].id] + ":" + conf;

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		top = std::max(top - 7, 0); // 把标签往上挪动，方便可视化
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	
}