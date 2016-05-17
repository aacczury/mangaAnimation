#ifndef _GRAPH_FILE_H_
#define _GRAPH_FILE_H_

#include <opencv2/core/core.hpp>

#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#ifndef _POINT2D_HASH_
#define _POINT2D_HASH_
namespace std{ // cv::Point unordered_set and unordered_map need a hash function
	template <> struct hash<cv::Point2d>{
		size_t operator()(cv::Point2d const &p) const{
			return 53 + std::hash<double>()(p.x) * 53 + std::hash<double>()(p.y);
		}
	};
}
#endif

class GraphFile
{
public:
	GraphFile(){};
	GraphFile(char *);

	std::unordered_set<cv::Point2d> graph_pnts;
	std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> graph;
	std::unordered_set<cv::Point2d> end_pnts;
	std::unordered_set<cv::Point2d> junction_pnts;
	std::vector<std::vector<cv::Point2d>> curves, sample_curves;
	std::vector<std::vector<cv::Point2d>> cycles, sample_cycles;
	std::vector<std::vector<unsigned int>> junction_cycles;
	std::unordered_map<cv::Point2d, std::vector<unsigned int>> pnt_to_curve;

	std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> junction_map;
	std::unordered_map<unsigned int, std::unordered_set<unsigned int>> path_map;

private:
	void read_file(char *);
	void build_curves();
	void build_pnt_to_curve();
	void build_path_map();

	std::vector<cv::Point2d> link_curve(cv::Point2d p, cv::Point2d &q, std::unordered_map<cv::Point2d, std::unordered_set<cv::Point2d>> &pnts_used_pnts);
	void find_cycle(std::unordered_map<cv::Point2d, bool> visited_pnts);
	void find_junction_cycle();

	std::vector<unsigned int> bt;
	void backtrack_cycle(cv::Point2d p, cv::Point2d q, unsigned int idx, unsigned int depth, unsigned int max_depth);
};

#endif