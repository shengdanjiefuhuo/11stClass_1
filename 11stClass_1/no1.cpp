#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int calcVisbalDft(cv::Mat srcMat, cv::Mat& magMat, cv::Mat& ph, double& normVal);                //���㸵��Ҷ�任���ӻ��ķ�ֵ��
int calcIdftImage(cv::Mat magMat, cv::Mat ph, double normVal, cv::Size srcSize, cv::Mat& dstMat);//idft
int selectPolygon(cv::Mat srcMat, cv::Mat& dstMat, bool DelLowFrq);                              //ѡ��Ҫ�����Ƶ�ʣ�����һ��mask
void on_mouse(int EVENT, int x, int y, int flags, void* userdata);                               //�����Ӧ�ص�����
void on_CircleRoi(int, void* userdata);                                                          //Բ��ROI����ص�����
void on_addImg(int, void*);                                                                      //ͼ���ϻص�����

std::vector<Point>  mousePoints;
Point points;
//ͼ���ϻص�����ȫ�ֱ���
int alpha = 5;          //addWeighted()��alpha����
int gamma = 0;          //addWeighted()��gamma����
cv::Mat addedMat;       //ͼ���Ͻ��
cv::Mat idft1, idft2;   //����Ҷ��任���

//Բ��ROI����ȫ�ֱ���
int radius = 16, r = 16;

/***************************************************************���㸵��Ҷ�任���ӻ��ķ�ֵ��************************************************************/
//����һ��ͼƬ������丵��Ҷ�任��Ŀ��ӻ��ķ�ֵ��
//ͬʱ�����λ�ף��ͻ�ԭ��һ��ʱ��ϵ���������ֵ
int calcVisbalDft(cv::Mat srcMat, cv::Mat& magMat, cv::Mat& ph, double& normVal)
{
	//��ͼ��ĳߴ���2��3��5��������ʱ����ɢ����Ҷ�任�ļ����ٶ���졣	
	//�������ͼ�����ѱ任�ߴ�
	int m = getOptimalDFTSize(srcMat.rows);
	int n = getOptimalDFTSize(srcMat.cols);

	Mat padded;
	//���³ߴ��ͼƬ���б�Ե��Ե���
	//�ѻҶ�ͼ��������Ͻ�,���ұߺ��±���չͼ��,��չ�������Ϊ0;
	copyMakeBorder(srcMat, padded, 0, m - srcMat.rows, 0, n - srcMat.cols, BORDER_CONSTANT, Scalar::all(0));

	//����һ������,�洢Ƶ��ת����float���͵ĸ���
	//planes[0]Ϊdft�任��ʵ����planes[1]Ϊ�鲿��phΪ��λ��magΪ��ֵ
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat mag = Mat_<float>(padded);

	//������λ��Mat_����ȷ�����������ͣ�����Ԫ��ʱ����Ҫ��ָ��Ԫ�����ͣ�
	ph = Mat_<float>(padded);
	Mat complexImg;
	//��ͨ��complexImg����ʵ�������鲿
	merge(planes, 2, complexImg);

	//���ϱߺϳɵ�mat���и���Ҷ�任,***֧��ԭ�ز���***,����Ҷ�任���Ϊ����.ͨ��1�����ʵ��,ͨ����������鲿
	dft(complexImg, complexImg);
	//�ѱ任��Ľ���ָ����mat,һ��ʵ��,һ���鲿,�����������
	split(complexImg, planes);

	//---------------�˲���Ŀ��Ϊ���õ���ʾ��ֵ---�����ָ�ԭͼʱ�����ٴ���һ��-------------------------
	magnitude(planes[0], planes[1], mag);//������mag
	phase(planes[0], planes[1], ph);//��λ��ph

	mag += Scalar::all(1);//�Է�ֵ��1
	//������ķ�ֵһ��ܴ󣬴ﵽ10^4,ͨ��û�а취��ͼ������ʾ��������Ҫ�������log��⡣
	log(mag, mag);

	//ȡ�����е����ֵ�����ں�����ԭʱȥ��һ��
	minMaxLoc(mag, 0, &normVal, 0, 0);

	//�޼�Ƶ��,���ͼ����л������������Ļ�,����Ƶ���ǲ��ԳƵ�,���Ҫ�޼�
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	ph = ph(Rect(0, 0, mag.cols & -2, mag.rows & -2));


	//��ʾ����Ƶ��ͼ
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	//������������Ϊ��׼����magͼ��ֳ��Ĳ���
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));   //���ϣ���û��Ϊq0,q1,q2,q3�����µ��ڴ�
	Mat q1(mag, Rect(cx, 0, cx, cy));  //����
	Mat q2(mag, Rect(0, cy, cx, cy));  //����
	Mat q3(mag, Rect(cx, cy, cx, cy)); //����
	//�������ޣ����mag���в���
	//���������½���
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	//���Ϻ����½���
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(mag, mag, 0, 1, NORM_MINMAX);
	//mag = mag * 255;
	//imwrite("ԭƵ��.jpg", mag);//δ��һ�����طֲ�����ͼ
	mag.copyTo(magMat);

	return 0;
}

/**********************************************************************����Ҷ��任***************************************************************************/
int calcIdftImage(cv::Mat magMat, cv::Mat ph, double normVal, cv::Size srcSize, cv::Mat& dstMat)
{
	cv::Mat mag = magMat.clone();
	//planes[0]Ϊdft�任��ʵ����planes[1]Ϊ�鲿��phΪ��λ�� plane_true=magΪ��ֵ
	Mat planes[] = { Mat::zeros(mag.size(), CV_32F), Mat::zeros(mag.size(), CV_32F) };

	Mat complexImg;
	//��ͨ��complexImg����ʵ�������鲿

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	//ǰ�����跴����һ�飬Ŀ����Ϊ����任��ԭͼ
	Mat q00(mag, Rect(0, 0, cx, cy));
	Mat q10(mag, Rect(cx, 0, cx, cy));
	Mat q20(mag, Rect(0, cy, cx, cy));
	Mat q30(mag, Rect(cx, cy, cx, cy));

	Mat tmp;
	//��������
	q00.copyTo(tmp);
	q30.copyTo(q00);
	tmp.copyTo(q30);
	q10.copyTo(tmp);
	q20.copyTo(q10);
	tmp.copyTo(q20);

	mag = mag * normVal;//����һ���ľ���ԭ 
	exp(mag, mag);		//��Ӧ��ǰ��ȥ����
	mag = mag - Scalar::all(1);//��Ӧǰ��+1
	polarToCart(mag, ph, planes[0], planes[1]);//�ɷ�����mag����λ��ph�ָ�ʵ��planes[0]���鲿planes[1]
	merge(planes, 2, complexImg);//��ʵ���鲿�ϲ�


	//-----------------------����Ҷ����任-----------------------------------
	Mat ifft;
	//����Ҷ��任
	idft(complexImg, ifft, DFT_REAL_OUTPUT);
	normalize(ifft, ifft, 0, 1, NORM_MINMAX);

	Rect rect(0, 0, srcSize.width, srcSize.height);
	cv::Mat dst;
	dst = ifft(rect);
	dst = dst * 255;

	dst.convertTo(dstMat, CV_8UC1);
	return 0;
}

/**********************************************************************************�����Ӧ����**********************************************************************************/
void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
	Mat hh;
	hh = *(Mat*)userdata;
	switch (EVENT)
	{
		//���������Ϣ
	case EVENT_LBUTTONDOWN://�������һ�Σ����ڵ�ǰ�������һ����
	{
		points.x = x;
		points.y = y;
		mousePoints.push_back(points);//���ص�ǰ���ѡ���������
		circle(hh, points, 4, Scalar(255, 255, 255), -1);//��Բ��������ʾ��ǰ��
		imshow("MaskRoi", hh);
	}
	break;
	}
}

/******************************************************************Բ��ROI����ص�����***************************************************************/
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

	contours.push_back(mousePoints);//���ص����������������У�����ֻ��һ��㣬��һ������
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
			drawContours(m, contours, 0, Scalar(0), -1);//�Է��صĵ�������������,ȥ��Ƶ
	}

	else
	{
		m = 0;
		if (contours[0].size() < 3)
			circle(m, p, radius, Scalar(1), -1);
		else
			drawContours(m, contours, 0, Scalar(1), -1);//�Է��صĵ�������������,ȥ��Ƶ
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
	//��ʼ��ʱ
	double start = static_cast<double>(getTickCount());

	cv::Mat magMat1, magMat2; //��ֵ
	cv::Mat ph1, ph2;         //��λ
	double normVal1, normVal2;//��һ��ϵ��

	//��ԭͼ
	cv::Mat src1 = imread("C:/Users/hp/cluo.jpg", 0);
	cv::Mat src2 = imread("C:/Users/hp/yaomin.jpg", 0);
	//�������ͼ
	calcVisbalDft(src1, magMat1, ph1, normVal1);
	calcVisbalDft(src2, magMat2, ph2, normVal2);
	//ȥ����ѡƵ��
	cv::Mat mask1, mask2;
	selectPolygon(magMat1, mask1, true);//ѡ��Ҫȥ���ĵ�Ƶ
	selectPolygon(magMat2, mask2, false);//ѡ��Ҫ�����ĵ�Ƶ
	cv::Mat proceMag1, proceMag2;
	proceMag1 = magMat1.mul(mask1);//ȥ����ѡƵ��
	proceMag2 = magMat2.mul(mask2);//ȥ����ѡƵ��
	imshow("src1�����ķ���ͼ", proceMag1);
	imshow("src2�����ķ���ͼ", proceMag2);

	//����idft
	calcIdftImage(proceMag1, ph1, normVal1, src1.size(), idft1);
	calcIdftImage(proceMag2, ph2, normVal2, src2.size(), idft2);
	imshow("idft1", idft1);
	imshow("idft2", idft2);

	//ͼ���ϻص�����
	namedWindow("dst");
	createTrackbar("alphaValue", "dst", &alpha, 9, on_addImg);
	createTrackbar("gammaValue", "dst", &gamma, 100, on_addImg);
	on_addImg(0, 0);

	waitKey(0);
	imwrite("���ͼ.jpg", addedMat);
	//������ʱ
	double time = ((double)getTickCount() - start) / getTickFrequency();
	//��ʾʱ��
	cout << "processing time:" << time / 1000 << "ms" << endl;

	return 0;
}