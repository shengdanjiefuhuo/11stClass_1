#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int calcVisbalDft(cv::Mat srcMat, cv::Mat& magMat, cv::Mat& ph, double& normVal);                //计算傅里叶变换可视化的幅值谱
int calcIdftImage(cv::Mat magMat, cv::Mat ph, double normVal, cv::Size srcSize, cv::Mat& dstMat);//idft
int selectPolygon(cv::Mat srcMat, cv::Mat& dstMat, bool DelLowFrq);                              //选择要处理的频率，返回一个mask
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);                               //鼠标响应回调函数
void on_CircleRoi(int, void* userdata);                                                          //圆形ROI区域回调函数
void on_addImg(int, void*);                                                                      //图像混合回调函数

std::vector<Point>  mousePoints;
Point points;
//图像混合回调函数全局变量
int alpha = 5;          //addWeighted()的alpha参数
int gamma = 0;          //addWeighted()的gamma参数
cv::Mat addedMat;       //图像混合结果
cv::Mat idft1, idft2;   //傅里叶逆变换结果

//圆形ROI区域全局变量
int radius = 16, r = 16;

/***************************************************************计算傅里叶变换可视化的幅值谱************************************************************/
//输入一张图片，输出其傅里叶变换后的可视化的幅值谱
//同时输出相位谱，和还原归一化时的系数，即最大值
int calcVisbalDft(cv::Mat srcMat, cv::Mat& magMat, cv::Mat& ph, double& normVal)
{
	//当图像的尺寸是2，3，5的整数倍时，离散傅里叶变换的计算速度最快。	
	//获得输入图像的最佳变换尺寸
	int m = getOptimalDFTSize(srcMat.rows);
	int n = getOptimalDFTSize(srcMat.cols);

	Mat padded;
	//对新尺寸的图片进行边缘边缘填充
	//把灰度图像放在左上角,在右边和下边扩展图像,扩展部分填充为0;
	copyMakeBorder(srcMat, padded, 0, m - srcMat.rows, 0, n - srcMat.cols, BORDER_CONSTANT, Scalar::all(0));

	//定义一个数组,存储频域转换成float类型的复数
	//planes[0]为dft变换的实部，planes[1]为虚部，ph为相位，mag为幅值
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat mag = Mat_<float>(padded);

	//保存相位（Mat_代表确定了数据类型，访问元素时不需要再指定元素类型）
	ph = Mat_<float>(padded);
	Mat complexImg;
	//多通道complexImg既有实部又有虚部
	merge(planes, 2, complexImg);

	//对上边合成的mat进行傅里叶变换,***支持原地操作***,傅里叶变换结果为复数.通道1存的是实部,通道二存的是虚部
	dft(complexImg, complexImg);
	//把变换后的结果分割到两个mat,一个实部,一个虚部,方便后续操作
	split(complexImg, planes);

	//---------------此部分目的为更好地显示幅值---后续恢复原图时反着再处理一遍-------------------------
	magnitude(planes[0], planes[1], mag);//幅度谱mag
	phase(planes[0], planes[1], ph);//相位谱ph

	mag += Scalar::all(1);//对幅值加1
	//计算出的幅值一般很大，达到10^4,通常没有办法在图像中显示出来，需要对其进行log求解。
	log(mag, mag);

	//取矩阵中的最大值，便于后续还原时去归一化
	minMaxLoc(mag, 0, &normVal, 0, 0);

	//修剪频谱,如果图像的行或者列是奇数的话,那其频谱是不对称的,因此要修剪
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	ph = ph(Rect(0, 0, mag.cols & -2, mag.rows & -2));


	//显示规则频谱图
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	//这里是以中心为标准，把mag图像分成四部分
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));   //左上，并没有为q0,q1,q2,q3开辟新的内存
	Mat q1(mag, Rect(cx, 0, cx, cy));  //右上
	Mat q2(mag, Rect(0, cy, cx, cy));  //左下
	Mat q3(mag, Rect(cx, cy, cx, cy)); //右下
	//交换象限，针对mag进行操作
	//左上与右下交换
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	//右上和左下交换
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(mag, mag, 0, 1, NORM_MINMAX);
	//mag = mag * 255;
	//imwrite("原频谱.jpg", mag);//未归一化的重分布幅度图
	mag.copyTo(magMat);

	return 0;
}

/**********************************************************************傅里叶逆变换***************************************************************************/
int calcIdftImage(cv::Mat magMat, cv::Mat ph, double normVal, cv::Size srcSize, cv::Mat& dstMat)
{
	cv::Mat mag = magMat.clone();
	//planes[0]为dft变换的实部，planes[1]为虚部，ph为相位， plane_true=mag为幅值
	Mat planes[] = { Mat::zeros(mag.size(), CV_32F), Mat::zeros(mag.size(), CV_32F) };

	Mat complexImg;
	//多通道complexImg既有实部又有虚部

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	//前述步骤反着来一遍，目的是为了逆变换回原图
	Mat q00(mag, Rect(0, 0, cx, cy));
	Mat q10(mag, Rect(cx, 0, cx, cy));
	Mat q20(mag, Rect(0, cy, cx, cy));
	Mat q30(mag, Rect(cx, cy, cx, cy));

	Mat tmp;
	//交换象限
	q00.copyTo(tmp);
	q30.copyTo(q00);
	tmp.copyTo(q30);
	q10.copyTo(tmp);
	q20.copyTo(q10);
	tmp.copyTo(q20);

	mag = mag * normVal;//将归一化的矩阵还原 
	exp(mag, mag);		//对应于前述去对数
	mag = mag - Scalar::all(1);//对应前述+1
	polarToCart(mag, ph, planes[0], planes[1]);//由幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
	merge(planes, 2, complexImg);//将实部虚部合并


	//-----------------------傅里叶的逆变换-----------------------------------
	Mat ifft;
	//傅里叶逆变换
	idft(complexImg, ifft, DFT_REAL_OUTPUT);
	normalize(ifft, ifft, 0, 1, NORM_MINMAX);

	Rect rect(0, 0, srcSize.width, srcSize.height);
	cv::Mat dst;
	dst = ifft(rect);
	dst = dst * 255;

	dst.convertTo(dstMat, CV_8UC1);
	return 0;
}

/**********************************************************************************鼠标响应函数**********************************************************************************/
void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
	Mat hh;
	hh = *(Mat*)userdata;
	switch (EVENT)
	{
		//左键按下消息
	case EVENT_LBUTTONDOWN://左键按下一次，就在当前坐标打下一个点
	{
		points.x = x;
		points.y = y;
		mousePoints.push_back(points);//返回当前鼠标选定点的坐标
		circle(hh, points, 4, Scalar(255, 255, 255), -1);//画圆，便于显示当前点
		imshow("MaskRoi", hh);
	}
	break;
	}
}

/******************************************************************圆形ROI区域回调函数***************************************************************/
void on_CircleRoi(int, void* userdata)
{

	Mat hh = (*(Mat*)userdata).clone();
	int cx = hh.cols / 2;
	int cy = hh.rows / 2;
	Point p(cx, cy);
	circle(hh, p, r, Scalar(0), 1);
	radius = r;
	imshow("MaskRoi", hh);
}

int selectPolygon(cv::Mat srcMat, cv::Mat& dstMat, bool DelLowFrq)
{
	vector<vector<Point>> contours;
	cv::Mat selectMat = srcMat.clone();

	namedWindow("MaskRoi");
	imshow("MaskRoi", selectMat);
	setMouseCallback("MaskRoi", on_mouse, &selectMat);
	createTrackbar("circle_r", "MaskRoi", &r, 100, on_CircleRoi, &selectMat);
	waitKey(0);
	destroyAllWindows();

	contours.push_back(mousePoints);//返回点向量到轮廓向量中，这里只有一组点，即一个轮廓
	mousePoints.clear();

	cv::Mat m = cv::Mat::zeros(selectMat.size(), CV_32F);
	int cx = m.cols / 2;
	int cy = m.rows / 2;
	Point p(cx, cy);

	if (DelLowFrq)
	{
		m = 1;
		if (contours[0].size() < 3)
			circle(m, p, radius, Scalar(0), -1);
		else
			drawContours(m, contours, 0, Scalar(0), -1);//以返回的点向量绘制轮廓,去低频
	}

	else
	{
		m = 0;
		if (contours[0].size() < 3)
			circle(m, p, radius, Scalar(1), -1);
		else
			drawContours(m, contours, 0, Scalar(1), -1);//以返回的点向量绘制轮廓,去高频
	}
	m.copyTo(dstMat);
	return 0;
}

void on_addImg(int, void*)
{
	double alphaValue = alpha ? alpha / 10.0 : 0.5;
	double betaValue = 1.0 - alphaValue;
	addWeighted(idft1, alphaValue, idft2, betaValue, (double)gamma, addedMat);

	imshow("dst", addedMat);
}

int main()
{
	//开始计时
	double start = static_cast<double>(getTickCount());

	cv::Mat magMat1, magMat2; //幅值
	cv::Mat ph1, ph2;         //相位
	double normVal1, normVal2;//归一化系数

	//读原图
	cv::Mat src1 = imread("C:/Users/hp/cluo.jpg", 0);
	cv::Mat src2 = imread("C:/Users/hp/yaomin.jpg", 0);
	//计算幅度图
	calcVisbalDft(src1, magMat1, ph1, normVal1);
	calcVisbalDft(src2, magMat2, ph2, normVal2);
	//去除所选频率
	cv::Mat mask1, mask2;
	selectPolygon(magMat1, mask1, true);//选择要去掉的低频
	selectPolygon(magMat2, mask2, false);//选择要保留的低频
	cv::Mat proceMag1, proceMag2;
	proceMag1 = magMat1.mul(mask1);//去掉所选频率
	proceMag2 = magMat2.mul(mask2);//去掉所选频率
	imshow("src1处理后的幅度图", proceMag1);
	imshow("src2处理后的幅度图", proceMag2);

	//计算idft
	calcIdftImage(proceMag1, ph1, normVal1, src1.size(), idft1);
	calcIdftImage(proceMag2, ph2, normVal2, src2.size(), idft2);
	imshow("idft1", idft1);
	imshow("idft2", idft2);

	//图像混合回调函数
	namedWindow("dst");
	createTrackbar("alphaValue", "dst", &alpha, 9, on_addImg);
	createTrackbar("gammaValue", "dst", &gamma, 100, on_addImg);
	on_addImg(0, 0);

	waitKey(0);
	imwrite("混合图.jpg", addedMat);
	//结束计时
	double time = ((double)getTickCount() - start) / getTickFrequency();
	//显示时间
	cout << "processing time:" << time / 1000 << "ms" << endl;

	return 0;
}