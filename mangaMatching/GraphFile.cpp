#include "GraphFile.h"


GraphFile::GraphFile(char *filename){
	read_file(filename);
	build_curves();
	build_pnt_to_curve();
	build_path_map();
	find_junction_cycle();
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
	std::unordered_map<cv::Point2d, bool> visited_pnts;
	for (const cv::Point2d &p : graph_pnts)	visited_pnts[p] = false;

	end_pnts.clear(), junction_pnts.clear();
	for (const cv::Point2d &p : graph_pnts){
		if (graph[p].size() == 1){
			end_pnts.insert(p);
			pnts_used_pnts[p] = graph[p];
			visited_pnts[p] = true;
		}
		if (graph[p].size() > 2){
			junction_pnts.insert(p);
			pnts_used_pnts[p] = graph[p];
			visited_pnts[p] = true;
		}
	}

	curves.clear();
	for (const cv::Point2d p : end_pnts){
		cv::Point2d q = *(graph[p].begin());
		if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
		curves.push_back(link_curve(p, q, pnts_used_pnts));
		for (const cv::Point2d r : curves[curves.size() - 1]) visited_pnts[r] = true;
	}

	for (const cv::Point2d p : junction_pnts){
		for (cv::Point2d q : graph[p]){
			if (pnts_used_pnts[p].find(q) == pnts_used_pnts[p].end()) continue;
			curves.push_back(link_curve(p, q, pnts_used_pnts));
			for (const cv::Point2d r : curves[curves.size() - 1]) visited_pnts[r] = true;
			junction_map[p].insert(q);
			junction_map[q].insert(p);
		}
	}
	find_cycle(visited_pnts);

	std::sort(curves.begin(), curves.end(), curve_length_cmp);
	printf("==> Building curves done\n");
	return;
}

void GraphFile::find_cycle(std::unordered_map<cv::Point2d, bool> visited_pnts){
	cycles.clear();
	for (const auto p : visited_pnts){ // pnt is 2 degree => cycle
		if (p.second == false){
			std::vector<cv::Point2d> cycle;
			cv::Point2d q = p.first, prev_q = cv::Point2d(-100, -100);
			do{
				visited_pnts[q] = true;
				cycle.push_back(q);
				std::unordered_set<cv::Point2d>::iterator qit = graph[q].begin();
				cv::Point2d q1 = *qit, q2 = *(++qit);
				if (prev_q == cv::Point2d(-100, -100) || prev_q != q1)
					prev_q = q, q = q1;
				else
					prev_q = q, q = q2;
			} while (q != p.first);
			cycles.push_back(cycle);
		}
	}
	return;
}

void GraphFile::backtrack_cycle(cv::Point2d p, cv::Point2d q, unsigned int idx, unsigned int depth, unsigned int max_depth){
	if (depth >= max_depth) return;
	if (depth && std::find(bt.begin(), bt.end(), idx) != bt.end()) return;
	if (depth && q != curves[idx][0] && q != curves[idx].back()) return;
	if (depth) q = q == curves[idx][0] ? curves[idx].back() : curves[idx][0];

	bt[depth] = idx;
	if (p == q){
		std::vector<unsigned int> jc(bt.begin(), bt.begin() + depth + 1);
		unsigned int min_i = std::min_element(jc.begin(), jc.end()) - jc.begin();
		if (jc[(min_i + jc.size() - 1) % jc.size()] > jc[(min_i + 1) % jc.size()]){
			std::reverse(jc.begin(), jc.end());
			min_i = jc.size() - 1 - min_i;
		}
		std::rotate(jc.begin(), jc.begin() + min_i, jc.end());
		if (std::find(junction_cycles.begin(), junction_cycles.end(), jc) == junction_cycles.end())
			junction_cycles.push_back(jc);
		return;
	};
	for (const auto next_idx : path_map[idx])
		if (!depth || next_idx != bt[depth - 1])
			backtrack_cycle(p, q, next_idx, depth + 1, max_depth);
	return;
}

void GraphFile::find_junction_cycle(){
	junction_cycles.clear();
	bt.resize(6);
	for (unsigned int i = 0; i < curves.size(); ++i){
		backtrack_cycle(curves[i][0], curves[i].back(), i, 0, 6);
		backtrack_cycle(curves[i].back(), curves[i][0], i, 0, 6);
	}
	return;
}

void GraphFile::build_pnt_to_curve(){
	pnt_to_curve.clear();
	for (unsigned int i = 0; i < curves.size(); ++i){
		pnt_to_curve[curves[i][0]].push_back(i);
		pnt_to_curve[curves[i][curves[i].size() - 1]].push_back(i);
	}
	return;
}

void GraphFile::build_path_map(){
	path_map.clear();
	for (unsigned int i = 0; i < curves.size(); ++i){
		path_map[i].insert(pnt_to_curve[curves[i][0]].begin(), pnt_to_curve[curves[i][0]].end());
		path_map[i].insert(pnt_to_curve[curves[i].back()].begin(), pnt_to_curve[curves[i].back()].end());
		path_map[i].erase(i);
	}
	return;
}

std::vector<cv::Point2d> GraphFile::link_curve(cv::Point2d p, cv::Point2d &q, std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> &pnts_used_pnts){
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