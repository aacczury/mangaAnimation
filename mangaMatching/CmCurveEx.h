#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef FALSE
#define FALSE               0
#endif

#ifndef TRUE
#define TRUE                1
#endif

#pragma once
using namespace std;
using namespace cv;

/************************************************************************/
/*  This software is developed by Ming-Ming Cheng.				        */
/*       Url: http://cg.cs.tsinghua.edu.cn/people/~cmm/                 */
/*  This software is free fro non-commercial use. In order to use this	*/
/*  software for academic use, you must cite the corresponding paper:	*/
/*      Ming-Ming Cheng, Curve Structure Extraction for Cartoon Images, */
/*		Proceedings of NCMT 2009, 1-8									*/	
/************************************************************************/

void hsv_2_rgb(int h, double s, double v, double &r, double &g, double &b);

class CmCurveEx
{
public:
	typedef struct CEdge{
		CEdge(int _index){index = _index; }
		~CEdge(void){}

		// Domains assigned during link();
		int index;    // Start from 0
		int pointNum; 
		Point start, end; 
		vector<Point> pnts; 
	}CEdge;

	CmCurveEx(const Mat& srcImg1f, float maxOrntDif = 0.25f * CV_PI);
	
	// Input kernel size for calculating derivatives, kSize should be 1, 3, 5 or 7
	const Mat& CalSecDer(int kSize = 5, float linkEndBound = 0.01f, float linkStartBound = 0.1f);
	const Mat& CalFirDer(int kSize = 5, float linkEndBound = 0.01f, float linkStartBound = 0.1f);
	const vector<CEdge>& Link(int shortRemoveBound = 3);
	void ReLink(float linkEndBound, float linkStartBound);
	void ReLink2(float linkEndBound, float linkStartBound, int shortRemoveBound);
	Mat GetOrntHSV()
	{
		double rgb[3];
		cv::Point pt;
		cv::Mat show3u = cv::Mat(m_img1f.size(), CV_8UC3, cv::Scalar(255, 255, 255));

		cv::Mat img_edge = GetEdge();

		for (pt.y=0 ; pt.y<show3u.rows ; pt.y++)
		{
			for (pt.x=0 ; pt.x<show3u.cols ; pt.x++)
			{
				float h, s, v;
				h = m_pOrnt1f.at<float>(pt) / M_PI * 180.0;
				v = img_edge.at<uchar>(pt);
				v /= 255.0;
				hsv_2_rgb(h, 1, v, rgb[0], rgb[1], rgb[2]);
				show3u.at<cv::Vec3b>(pt) = cv::Vec3b(rgb[0]*255.0, rgb[1]*255.0, rgb[2]*255.0);
			}
		}

		cv::flip(show3u, show3u, 0);
		cv::imwrite("curve.png", show3u);
		exit(0);
		return show3u;
	}
	// Get data pointers
	Mat GetEdge() 
	{ 
		Mat img_gray;
		vector<PntImp>::iterator it;

		img_gray = cv::Mat(m_img1f.size(), CV_8UC1, cv::Scalar(0));
		for (it = m_StartPnt.begin(); it != m_StartPnt.end(); it++)
		{
			img_gray.at<uchar>(it->second) = 255;
		}
		return img_gray;
	}
	const Mat& GetDer(){ return m_pDer1f; }
	const Mat& GetLineIdx() { return m_pLabel1i; } // Edge index start from 1
	const Mat& GetNextMap() { return m_pNext1i; }
	const Mat& GetOrnt() { return m_pOrnt1f; }
	const vector<CEdge>& GetEdges() {return m_vEdge;}

	static const int IND_BG = 0xffffffff, IND_NMS = 0xfffffffe, IND_SR = 0xfffffffd; // Background, Non Maximal Suppress and Short Remove

	static void Demo(const Mat &img1u, bool isCartoon);

protected:
	const Mat &m_img1f; // Input image

	Mat m_pDer1f;   // First or secondary derivatives. 32FC1
	Mat m_pOrnt1f;  // Line orientation. 32FC1
	Mat m_pLabel1i;  // Line index, 32SC1.
	Mat m_pNext1i;   // Next point 8-direction index, [0, 1, ...,  7], 32SC1

	// Will be used for link process
	typedef pair<float, Point> PntImp;
	vector<PntImp> m_StartPnt;
	vector<CEdge> m_vEdge;
	static bool linePointGreater (const PntImp& e1, const PntImp& e2 ) {return e1.first > e2.first;};

	int m_h, m_w; // Image size	
	int m_kSize; // Smooth kernel size: 1, 3, 5, 7
	float m_maxAngDif; // maximal allowed angle difference in a curve

	void NoneMaximalSuppress(float linkEndBound, float linkStartBound);
	void findEdge(Point seed, CEdge& crtEdge, bool isBackWard);
	bool goNext(Point &pnt, float& ornt, CEdge& crtEdge, int orntInd, bool isBackward);
	bool jumpNext(Point &pnt, float& ornt, CEdge& crtEdge, int orntInd, bool isBackward);

	/* Compute the eigenvalues and eigenvectors of the Hessian matrix given by
	dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
	their absolute values. */
	static void compute_eigenvals(double dfdrr, double dfdrc, double dfdcc, double eigval[2], double eigvec[2][2]);
	
	static inline float angle(float ornt1, float orn2);
	static inline void refreshOrnt(float& ornt, float& newOrnt);
};

typedef CmCurveEx::CEdge CmEdge;
typedef vector<CmEdge> CmEdges;
