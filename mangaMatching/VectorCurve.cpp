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


void VectorCurve::Link(int shortRemoveBound /* = 3 */){
	CV_Assert(m_pDer1f.data != NULL && m_pLabel1i.data != NULL);

	sort(m_StartPnt.begin(), m_StartPnt.end(), linePointGreater);

	m_pNext1i = -1;
	curves.clear();
	for (vector<PntImp>::iterator it = m_StartPnt.begin(); it != m_StartPnt.end(); it++)
	{
		Point pnt = it->second;
		if (m_pLabel1i.at<int>(pnt) != IND_NMS)
			continue;
		m_pLabel1i.at<int>(pnt) = curves.size();
		std::vector<cv::Point> curve;
		curve.push_back(pnt);

		std::vector<cv::Point> forward_edge;
		findEdge(pnt, forward_edge, FALSE, curves.size());
		std::vector<cv::Point> backward_edge;
		findEdge(pnt, backward_edge, TRUE, curves.size());
		std::reverse(backward_edge.begin(), backward_edge.end());

		curve.insert(curve.begin(), backward_edge.begin(), backward_edge.end());
		curve.insert(curve.end(), forward_edge.begin(), forward_edge.end());

		if (curve.size() <= shortRemoveBound) {
			for (size_t i = 0; i < curve.size(); ++i)
				m_pLabel1i.at<int>(curve[i]) = IND_SR;
		}
		else{
			curves.push_back(curve);
		}
	}
	printf("=> Finding %d curve done.\n", curves.size());
	return;
}


bool curve_size_cmp(std::vector<cv::Point> &a, std::vector<cv::Point> &b){
	return a.size() > b.size();
}
void VectorCurve::remove_duplication(unsigned int width, unsigned height, unsigned short thickness){
	std::sort(curves.begin(), curves.end(), curve_size_cmp);
	std::vector<bool> is_remove;
	is_remove.resize(curves.size());
	for (unsigned int i = 0; i < curves.size(); ++i){
		cv::Mat ROI = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
		for (unsigned int j = 1; j < curves[i].size(); ++j)
			cv::line(ROI, curves[i][j - 1], curves[i][j], cv::Scalar(1), thickness);

		for (unsigned int j = i + 1; j < curves.size(); ++j){

			unsigned int useless_pnt = 0;
			for (unsigned int k = 0; k < curves[j].size(); ++k){
				if (ref_Mat_val(ROI, uchar(0), curves[j][k]))
					useless_pnt++;
			}

			if (useless_pnt == curves[j].size())
				is_remove[j] = true;
		}
	}
	unsigned int curr_curves_num = curves.size();
	remove_curves(is_remove);
	printf("=> Removing %d curve done.\n", curr_curves_num - curves.size());
	return;
}

void VectorCurve::link_curves(){
	for (unsigned int i = 0; i < curves.size(); ++i){
		curves_pnts.insert(curves[i][0]);
		for (unsigned int j = 1; j < curves[i].size(); ++j){
			curves_pnts.insert(curves[i][j]);
			topol[curves[i][j - 1]].insert(curves[i][j]);
			topol[curves[i][j]].insert(curves[i][j - 1]);
		}
	}
	printf("=> Link curves done.\n");
	return;
}

void VectorCurve::link_adjacent(){ // maybe need speed up
	for (std::unordered_set<cv::Point>::iterator pit = curves_pnts.begin(); pit != curves_pnts.end(); ++pit){
		std::unordered_set<cv::Point>::iterator qit = pit;
		for (++qit; qit != curves_pnts.end(); ++qit){
			cv::Point dis = *pit - *qit;
			dis = cv::Point(abs(dis.x), abs(dis.y));
			if (dis == cv::Point(1, 0) || dis == cv::Point(1, 1) || dis == cv::Point(0, 1)){
				topol[*pit].insert(*qit);
				topol[*qit].insert(*pit);
			}
		}
	}
	printf("=> Link adjacent done.\n");
	return;
}

void VectorCurve::relink_4_degree(){
	for (const cv::Point &p : curves_pnts){
		if (topol[p].size() == 4){
			unsigned short is_through[8] = { 0 };
			// 5 6 7
			// 4   0
			// 3 2 1
			int line_case = 0;
			bool other_straight_line = false;
			if (topol[p].find(p + DIRECTION8[1]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[5]) != topol[p].end()){
				is_through[1] += 2, is_through[5] += 2;
				line_case += 1;
			}
			if (topol[p].find(p + DIRECTION8[2]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[6]) != topol[p].end()){
				is_through[2] += 2, is_through[6] += 2;
				line_case += 2;
			}
			if (topol[p].find(p + DIRECTION8[3]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[7]) != topol[p].end()){
				is_through[3] += 2, is_through[7] += 2;
				line_case += 4;
			}
			if (topol[p].find(p + DIRECTION8[4]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[0]) != topol[p].end()){
				is_through[4] += 2, is_through[0] += 2;
				line_case += 8;
			}

			if (topol[p].find(p + DIRECTION8[1]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[2]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[3]) != topol[p].end()){
				is_through[1] ++, is_through[2] ++, is_through[3] ++;
				other_straight_line = true;
			}
			else if (topol[p].find(p + DIRECTION8[3]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[4]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[5]) != topol[p].end()){
				is_through[3] ++, is_through[4] ++, is_through[5] ++;
				other_straight_line = true;
			}
			else if (topol[p].find(p + DIRECTION8[5]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[6]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[7]) != topol[p].end()){
				is_through[5] ++, is_through[6] ++, is_through[7] ++;
				other_straight_line = true;
			}
			else if (topol[p].find(p + DIRECTION8[7]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[0]) != topol[p].end() &&
				topol[p].find(p + DIRECTION8[1]) != topol[p].end()){
				is_through[7] ++, is_through[0] ++, is_through[1] ++;
				other_straight_line = true;
			}

			if (line_case > 0 && other_straight_line){
				for (unsigned short i = 0; i < 8; ++i){
					if (is_through[i] == 1) topol[p + DIRECTION8[i]].erase(p);
				}
				switch (line_case)
				{
				case 1: topol[p].clear(); topol[p].insert(p + DIRECTION8[1]); topol[p].insert(p + DIRECTION8[5]); break;
				case 2: topol[p].clear(); topol[p].insert(p + DIRECTION8[2]); topol[p].insert(p + DIRECTION8[6]); break;
				case 4: topol[p].clear(); topol[p].insert(p + DIRECTION8[3]); topol[p].insert(p + DIRECTION8[7]); break;
				case 8: topol[p].clear(); topol[p].insert(p + DIRECTION8[4]); topol[p].insert(p + DIRECTION8[0]); break;
				}
			}
			else{
				printf("==> What a terrible 4 degree !!\n");
				if (topol[p].find(p + DIRECTION8[5]) != topol[p].end()) printf("O"); else printf("X");
				if (topol[p].find(p + DIRECTION8[6]) != topol[p].end()) printf("O"); else printf("X");
				if (topol[p].find(p + DIRECTION8[7]) != topol[p].end()) printf("O"); else printf("X");
				printf("\n");
				if (topol[p].find(p + DIRECTION8[4]) != topol[p].end()) printf("O"); else printf("X");
				printf("O");
				if (topol[p].find(p + DIRECTION8[0]) != topol[p].end()) printf("O"); else printf("X");
				printf("\n");
				if (topol[p].find(p + DIRECTION8[3]) != topol[p].end()) printf("O"); else printf("X");
				if (topol[p].find(p + DIRECTION8[2]) != topol[p].end()) printf("O"); else printf("X");
				if (topol[p].find(p + DIRECTION8[1]) != topol[p].end()) printf("O"); else printf("X");
				printf("\n");
			}
		}
	}
	printf("=> Relinking 4 degree done.\n");
	return;
}

void VectorCurve::relink_3_degree(){
	for (const cv::Point &p : curves_pnts){
		if (topol[p].size() == 3){
			std::vector<cv::Point> ab, c;
			for (const cv::Point &q : topol[p]){
				cv::Point dis = q - p;
				dis = cv::Point(abs(dis.x), abs(dis.y));
				if ((dis.x + dis.y) == 1 && topol[q].size() == 3) ab.push_back(q);
				else c.push_back(q);
			}
			if (ab.size() == 2 && c.size() == 1){
				topol[ab[0]].erase(c[0]), topol[ab[0]].erase(ab[1]);
				topol[ab[1]].erase(c[0]), topol[ab[1]].erase(ab[0]);
			}
		}
	}
	for (const cv::Point &p : curves_pnts){
		if (topol[p].size() == 3){
			std::vector<cv::Point> ab, c;
			for (const cv::Point &q : topol[p]){
				if (topol[q].size() == 3) ab.push_back(q);
				else c.push_back(q);
			}
			if (ab.size() == 2 && c.size() == 1){
				topol[ab[0]].erase(c[0]), topol[ab[0]].erase(ab[1]);
				topol[ab[1]].erase(c[0]), topol[ab[1]].erase(ab[0]);
			}
		}
	}
	printf("=> Relinking 3 degree done.\n");
	return;
}

void VectorCurve::relink_curves(){
	for (const cv::Point &p : curves_pnts){
		if (topol[p].size() > 2){
			junction_pnts.insert(p);
			printf("%d: (%d, %d)\n", topol[p].size(), p.x, p.y);
		}
		else if (topol[p].size() == 1)
			end_pnts.insert(p);
	}
}

std::vector<std::vector<cv::Point>> VectorCurve::get_curves(){
	return curves;
}

std::unordered_set<cv::Point> VectorCurve::get_curves_pnts(){
	return curves_pnts;
}

std::unordered_map<cv::Point, unordered_set<cv::Point>> VectorCurve::get_topol(){
	return topol;
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

void VectorCurve::remove_curves(std::vector<bool> is_remove){
	assert(is_remove.size() == curves.size());
	std::vector<std::vector<cv::Point>> tmp_curves = curves;
	curves.clear();
	for (unsigned int i = 0; i < tmp_curves.size(); ++i)
		if (!is_remove[i]) curves.push_back(tmp_curves[i]);
	return;
}