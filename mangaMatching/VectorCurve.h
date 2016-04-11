#ifndef _VECTORCURVE_H_
#define _VECTORCURVE_H_

#include "CmCurveEx.h"

class VectorCurve : public CmCurveEx{
public:
	typedef struct endPoint{
		endPoint(bool s) : state(s){ pnt = cv::Point(-1, -1); };
		endPoint(bool s, cv::Point p) : state(s), pnt(p){};

		bool state;
		cv::Point pnt;
	}endPoint;

	VectorCurve(const Mat& srcImg1f, float maxOrntDif = 0.25f * CV_PI)
		: CmCurveEx(srcImg1f, maxOrntDif){
	}

	std::vector<std::vector<cv::Point>> Link(int shortRemoveBound = 3);
	void findEdge(cv::Point seed, std::vector<cv::Point> &edge, bool isBackWard, int index);
	bool findNext(cv::Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index, int dir, endPoint &ep);
	endPoint goNext(cv::Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index);
	endPoint jumpNext(cv::Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index);
};

typedef VectorCurve::endPoint endPoint;

#endif