#include "VectorCurve.h"

Point static const DIRECTION4[4] = {
	Point(1, 0), //Direction 0
	Point(0, 1), //Direction 1
	Point(-1, 0), //Direction 2
	Point(0, -1), //Direction 3
};  //format: {dx, dy}

Point static const DIRECTION8[8] = {
	Point(1, 0), //Direction 0
	Point(1, 1), //Direction 1 
	Point(0, 1), //Direction 2
	Point(-1, 1), //Direction 3
	Point(-1, 0), //Direction 4
	Point(-1, -1), //Direction 5
	Point(0, -1), //Direction 6
	Point(1, -1)  //Direction 7
};  //format: {dx, dy}

Point static const DIRECTION16[16] = {
	Point(2, 0), //Direction 0
	Point(2, 1), //Direction 1 
	Point(2, 2), //Direction 2
	Point(1, 2), //Direction 3
	Point(0, 2), //Direction 4
	Point(-1, 2), //Direction 5
	Point(-2, 2), //Direction 6
	Point(-2, 1), //Direction 7
	Point(-2, 0), //Direction 8
	Point(-2, -1), //Direction 9 
	Point(-2, -2), //Direction 10
	Point(-1, -2), //Direction 11
	Point(0, -2), //Direction 12
	Point(1, -2), //Direction 13
	Point(2, -2), //Direction 14
	Point(2, -1)  //Direction 15
}; //format: {dx, dy}

float static const DRT_ANGLE[8] = {
	0.000000f,
	0.785398f,
	1.570796f,
	2.356194f,
	3.141593f,
	3.926991f,
	4.712389f,
	5.497787f
};

#define DIR08 0
#define DIR16 1

float static const PI_FLOAT = 3.1415926535897932384626433832795f;
float static const PI2 = PI_FLOAT * 2.0f;
float static const PI_HALF = PI_FLOAT * 0.5f;

const double EPS = 1e-8;		// Epsilon (zero value)
#define CHK_IND(p) ((p).x >= 0 && (p).x < m_w && (p).y >= 0 && (p).y < m_h)

template<typename T> inline int CmSgn(T number) { if (abs(number) < EPS) return 0; return number > 0 ? 1 : -1; }

float const static PI_QUARTER = PI_FLOAT * 0.25f;
float const static PI_EIGHTH = PI_FLOAT * 0.125f;


std::vector<std::vector<cv::Point>> VectorCurve::Link(int shortRemoveBound /* = 3 */){
	CV_Assert(m_pDer1f.data != NULL && m_pLabel1i.data != NULL);

	sort(m_StartPnt.begin(), m_StartPnt.end(), linePointGreater);

	m_pNext1i = -1;
	std::vector<std::vector<cv::Point>> edges;
	for (vector<PntImp>::iterator it = m_StartPnt.begin(); it != m_StartPnt.end(); it++)
	{
		Point pnt = it->second;
		if (m_pLabel1i.at<int>(pnt) != IND_NMS)
			continue;
		m_pLabel1i.at<int>(pnt) = edges.size();
		std::vector<cv::Point> edge;
		edge.push_back(pnt);

		std::vector<cv::Point> forward_edge;
		findEdge(pnt, forward_edge, FALSE, edges.size());
		std::vector<cv::Point> backward_edge;
		findEdge(pnt, backward_edge, TRUE, edges.size());
		std::reverse(backward_edge.begin(), backward_edge.end());

		edge.insert(edge.begin(), backward_edge.begin(), backward_edge.end());
		edge.insert(edge.end(), forward_edge.begin(), forward_edge.end());

		if (edge.size() <= shortRemoveBound) {
			for (size_t i = 0; i < edge.size(); ++i)
				m_pLabel1i.at<int>(edge[i]) = IND_SR;
		}
		else{
			edges.push_back(edge);
		}
	}

	return edges;
}

void VectorCurve::findEdge(Point seed, std::vector<cv::Point> &edge, bool isBackWard, int index){
	Point pnt = seed;

	float ornt = m_pOrnt1f.at<float>(pnt);
	if (isBackWard){
		ornt += PI_FLOAT;
		if (ornt >= PI2)
			ornt -= PI2;
	}

	int orntInd, nextInd1, nextInd2;
	while (true) {
		endPoint ep(false), curr_ep(false);
		/*************按照優先及尋找下一個點，方向差異較大不加入**************/
		//最優DIRECTION16
		orntInd = int(ornt / PI_EIGHTH + 0.5f) % 16;
		if (findNext(pnt, ornt, edge, orntInd, index, DIR16, ep)) continue;
		//最優DIRECTION8
		orntInd = int(ornt / PI_QUARTER + 0.5f) % 8;
		if (findNext(pnt, ornt, edge, orntInd, index, DIR08, ep)) continue;
		//次優DIRECTION16
		orntInd = int(ornt / PI_EIGHTH + 0.5f) % 16;
		nextInd1 = (orntInd + 1) % 16;
		nextInd2 = (orntInd + 15) % 16;
		if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR16, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR16, ep)) continue;
		}
		else{//另一個DIRECTION16
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR16, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR16, ep)) continue;
		}
		//次優DIRECTION8
		orntInd = int(ornt / PI_QUARTER + 0.5f) % 8;
		nextInd1 = (orntInd + 1) % 8;
		nextInd2 = (orntInd + 7) % 8;
		if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR08, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR08, ep)) continue;
		}
		else{//另一個DIRECTION8
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR08, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR08, ep)) continue;
		}


		/*************按照優先級尋找下一個點, 方向差異較大也加入**************/
		//最優DIRECTION16
		orntInd = int(ornt / PI_EIGHTH + 0.5f) % 16;
		if (findNext(pnt, ornt, edge, orntInd, index, DIR16, ep)) continue;
		//最優DIRECTION8
		orntInd = int(ornt / PI_QUARTER + 0.5f) % 8;
		if (findNext(pnt, ornt, edge, orntInd, index, DIR08, ep)) continue;
		//次優DIRECTION16
		orntInd = int(ornt / PI_EIGHTH + 0.5f) % 16;
		nextInd1 = (orntInd + 1) % 16;
		nextInd2 = (orntInd + 15) % 16;
		if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR16, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR16, ep)) continue;
		}
		else{//另一個DIRECTION16
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR16, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR16, ep)) continue;
		}
		//次優DIRECTION8
		orntInd = int(ornt / PI_QUARTER + 0.5f) % 8;
		nextInd1 = (orntInd + 1) % 8;
		nextInd2 = (orntInd + 7) % 8;
		if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR08, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR08, ep)) continue;
		}
		else{//另一個DIRECTION8
			if (findNext(pnt, ornt, edge, nextInd2, index, DIR08, ep)) continue;
			if (findNext(pnt, ornt, edge, nextInd1, index, DIR08, ep)) continue;
		}

		if (ep.state) edge.push_back(ep.pnt);
		break;//如果ornt附近的三個方向上都沒有的話, 結束尋找
	}
}

bool VectorCurve::findNext(Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index, int dir, endPoint &ep){
	endPoint curr_ep(false);
	if (dir == DIR08)
		curr_ep = goNext(pnt, ornt, edge, orntInd, index);
	else if (dir == DIR16)
		curr_ep = jumpNext(pnt, ornt, edge, orntInd, index);
	else
		return false;

	if (curr_ep.state)
		return true;
	else if (curr_ep.pnt != cv::Point(-1, -1) && !ep.state){
		ep.pnt = curr_ep.pnt;
		ep.state = true;
	}
	return false;
}

endPoint VectorCurve::goNext(Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index){
	Point next_pnt = pnt + DIRECTION8[orntInd];
	int &label = m_pLabel1i.at<int>(next_pnt);

	if (CHK_IND(next_pnt) && (label == IND_NMS || label == IND_SR || label >= 0)) {
		if (angle(ornt, m_pOrnt1f.at<float>(next_pnt)) > m_maxAngDif)
			return endPoint(0);
		if (label >= 0)
			return endPoint(0, next_pnt);
		else
			label = index;
		edge.push_back(next_pnt);

		refreshOrnt(ornt, m_pOrnt1f.at<float>(next_pnt));
		pnt = next_pnt;

		return endPoint(1);
	}
	return endPoint(0);
}

endPoint VectorCurve::jumpNext(Point &pnt, float &ornt, std::vector<cv::Point> &edge, int orntInd, int index){
	Point pnt2 = pnt + DIRECTION16[orntInd];

	if (CHK_IND(pnt2) && (m_pLabel1i.at<int>(pnt2) <= IND_NMS || m_pLabel1i.at<int>(pnt2) >= 0)) {
		if (angle(ornt, m_pOrnt1f.at<float>(pnt2)) > m_maxAngDif)
			return endPoint(0);

		// DIRECTION16方向上的orntInd相當於DIRECTION8方向上兩個orntInd1,orntInd2
		// 的疊加,滿足orntInd = orntInd1 + orntInd2.此處優先選擇使得組合上的點具
		// IND_NMS標記的方向組合。(orntInd1,orntInd2在floor(orntInd/2)和
		// ceil(orntInd/2)中選擇
		int orntInd1 = orntInd >> 1, orntInd2;
		Point pnt1 = pnt + DIRECTION8[orntInd1];
		if (m_pLabel1i.at<int>(pnt1) >= IND_BG && orntInd % 2) {
			orntInd1 = ((orntInd + 1) >> 1) % 8;
			pnt1 = pnt + DIRECTION8[orntInd1];
		}

		int &lineIdx1 = m_pLabel1i.at<int>(pnt1);
		if (lineIdx1 != -1)
			return endPoint(0, pnt1);
		else
			lineIdx1 = index;

		if (m_pLabel1i.at<int>(pnt2) >= 0)
			return endPoint(0, pnt2);
		else
			m_pLabel1i.at<int>(pnt2) = index;

		orntInd2 = orntInd - orntInd1;
		orntInd2 %= 8;

		edge.push_back(pnt1);
		edge.push_back(pnt2);

		refreshOrnt(ornt, m_pOrnt1f.at<float>(pnt1));
		refreshOrnt(ornt, m_pOrnt1f.at<float>(pnt2));
		pnt = pnt2;

		return endPoint(1);
	}
	return endPoint(0);
}