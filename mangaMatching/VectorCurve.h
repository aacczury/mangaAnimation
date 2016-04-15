#ifndef _VECTORCURVE_H_
#define _VECTORCURVE_H_

#include "CmCurveEx.h"
#include <unordered_map>
#include <unordered_set>

#ifndef _POINT_HASH_
#define _POINT_HASH_
namespace std{ // cv::Point unordered_set and unordered_map need a hash function
	template <> struct hash<cv::Point>{
		size_t operator()(cv::Point const &p) const{
			return 53 + std::hash<int>()(p.x) * 53 + std::hash<int>()(p.y);
		}
	};
}
#endif

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

	void Link(int shortRemoveBound = 3);
	void remove_duplication(unsigned int width, unsigned height, unsigned short thickness = 2);
	void link_curves();
	void link_adjacent();
	void relink_4_degree();
	void relink_3_degree();
	void relink_curves();

	std::vector<std::vector<cv::Point>> get_curves();
	std::unordered_set<cv::Point> get_curves_pnts();
	std::unordered_map<cv::Point, unordered_set<cv::Point>> get_topol();

protected:
	std::vector<std::vector<cv::Point>> curves;
	std::unordered_set<cv::Point> curves_pnts;
	std::unordered_map<cv::Point, unordered_set<cv::Point>> topol;
	std::unordered_set<cv::Point> junction_pnts, end_pnts;
	std::vector<std::vector<cv::Point>> topol_curves;

	void findEdge(cv::Point seed, std::vector<cv::Point> &edge, bool isBackWard, int index);
	bool findNext(cv::Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index, int dir, endPoint &ep);
	endPoint goNext(cv::Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index);
	endPoint jumpNext(cv::Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index);
	void remove_curves(std::vector<bool> is_remove);

	template<typename T>
	// type:Mat type ex: uchar(0), i: row, j: col, c: channel
	T &ref_Mat_val(cv::Mat &m, T type, int i, int j, int c = 0){
		return ((T *)m.data)[(i * m.cols + j) * m.channels() + c];
	}
	template<typename T>
	// type:Mat type ex: uchar(0), i: row, j: col, c: channel
	T &ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c = 0){
		return ((T *)m.data)[(p.y * m.cols + p.x) * m.channels() + c];
	}
};

typedef VectorCurve::endPoint endPoint;

#endif