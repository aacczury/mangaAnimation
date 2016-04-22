#include "GraphFile.h"


GraphFile::GraphFile(char *filename){
	read_file(filename);
	build_curves();
	build_pnt_to_curve();
}

void GraphFile::read_file(char *filename){
	FILE *graph_file = fopen(filename, "r");
	std::vector<cv::Point2d> graph_pnts_vector;
	graph_pnts_vector.clear(), graph_pnts.clear(), graph.clear();

	unsigned int pnts_size;
	fscanf(graph_file, "%u\n", &pnts_size);
	for (unsigned int i = 0; i < pnts_size; ++i){
		cv::Point2d p;
		fscanf(graph_file, "V %lf %lf\n", &p.x, &p.y);
		graph_pnts_vector.push_back(p);
		graph_pnts.insert(p);
	}

	unsigned int edges_size;
	fscanf(graph_file, "%u\n", &edges_size);
	for (unsigned int i = 0; i < edges_size; ++i){
		unsigned int p, q;
		fscanf(graph_file, "S %u %u\n", &p, &q);
		graph[graph_pnts_vector[p]].insert(graph_pnts_vector[q]);
		graph[graph_pnts_vector[q]].insert(graph_pnts_vector[p]);
	}

	printf("=> Reading %s done.\n", filename);
	return;
}


bool curve_length_cmp(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b){
	double a_length = 0, b_length = 0;
	for (unsigned int i = 1; i < a.size(); ++i) a_length += cv::norm(a[i] - a[i - 1]);
	for (unsigned int i = 1; i < b.size(); ++i) b_length += cv::norm(b[i] - b[i - 1]);
	return a_length > b_length;
}
void GraphFile::build_curves(){
	std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> pnts_used_pnts;

	end_pnts.clear(), junction_pnts.clear();
	for (const cv::Point2d &p : graph_pnts){
		if (graph[p].size() == 1){
			end_pnts.insert(p);
			pnts_used_pnts[p] = graph[p];
		}
		if (graph[p].size() > 2){
			junction_pnts.insert(p);
			pnts_used_pnts[p] = graph[p];
		}
	}

	curves.clear();
	for (const cv::Point2d p : end_pnts){
		cv::Point2d q = *(graph[p].begin());
		if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
		curves.push_back(link_curve(p, q, pnts_used_pnts));
	}

	for (const cv::Point2d p : junction_pnts){
		for (const cv::Point2d q : graph[p]){
			if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
			curves.push_back(link_curve(p, q, pnts_used_pnts));
		}
	}

	std::sort(curves.begin(), curves.end(), curve_length_cmp);
	printf("==> Building curves done\n");
	return;
}

void GraphFile::build_pnt_to_curve(){
	pnt_to_curve.clear();
	for (unsigned int i = 0; i < curves.size(); ++i){
		pnt_to_curve[curves[i][0]].push_back(i);
		pnt_to_curve[curves[i][curves[i].size() - 1]].push_back(i);
	}
	printf("==> Building pnt to curve done\n");
	return;
}


std::vector<cv::Point2d> GraphFile::link_curve(cv::Point2d p, cv::Point2d q, std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> &pnts_used_pnts){
	std::vector<cv::Point2d> curve;
	curve.push_back(p);
	pnts_used_pnts[p].erase(q);
	while (end_pnts.find(q) == end_pnts.end() &&
		junction_pnts.find(q) == junction_pnts.end()){
		curve.push_back(q);
		for (const cv::Point2d &r : graph[q]){
			if (p != r){
				p = q;
				q = r;
				break;
			}
		}
	}
	pnts_used_pnts[q].erase(curve[curve.size() - 1]);
	curve.push_back(q);
	return curve;
}