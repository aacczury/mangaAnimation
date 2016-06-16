#include "mangaShow.h"

bool point2d_vector_size_cmp(const std::vector<cv::Point2d> &a, const std::vector<cv::Point2d> &b){ return a.size() > b.size(); }
bool p_x_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.x < b.x; }
bool p_y_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.y < b.y; }

int is_cur[12] = { 0, 1, 1, 0, 2, 3, 3, 1, 1, 1, 1, 1 };
double graph_sample = 0.005;
double sample_sample = 0.005;
double max_r_thresh = 0.5;
double scale_max = 1.4;
double scale_inter = 0.4;

mangaShow::mangaShow(){
	img_read = img_show = canvas = cv::Mat();
	curves_color.clear();
	curves_drawable.clear();
	mangaFace_CD.clear();
	sampleFace_CD.clear();
	notable.clear();
}

void mangaShow::read_img(char *filename){
	img_read = cv::imread(filename, CV_LOAD_IMAGE_COLOR); // need exception handling
	img_show = img_read.clone();
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));

	char ostr[100];
	sprintf(ostr, "ansicon -E [44mReading:[0m ");
	system(ostr);
	printf("%s done.\n", filename);
}

void mangaShow::read_graph(char *filename, int g_s){
	if (g_s == MANGA_FACE){
		mangaFace = GraphFile(filename);
		for (unsigned int i = 0; i < mangaFace.curves.size(); ++i){
			mangaFace_CD.push_back(CurveDescriptor(mangaFace.curves[i], graph_sample, 3.0, true));
			mangaFace.sample_curves.push_back(mangaFace_CD[i].curve);
		}
		for (unsigned int i = 0; i < mangaFace.cycles.size(); ++i){
			mangaFace.sample_cycles.push_back(CurveDescriptor(mangaFace.cycles[i], graph_sample, 3.0, false).curve);
		}
		assert(mangaFace.sample_curves.size() == mangaFace.curves.size());

		char ostr[100];
		sprintf(ostr, "ansicon -e [41mDebug:[0m curves size: %u, sample curves size: %u", mangaFace.curves.size(), mangaFace.sample_curves.size());
		system(ostr);
	}
	else if (g_s == SAMPLE_FACE){
		sampleFace = GraphFile(filename);
		for (unsigned int i = 0; i < sampleFace.curves.size(); ++i){
			sampleFace_CD.push_back(CurveDescriptor(sampleFace.curves[i], sample_sample, 3.0, true));
			sampleFace.sample_curves.push_back(sampleFace_CD.back().curve);
			prim_is_open.push_back(true);
		}
		for (unsigned int i = 0; i < sampleFace.cycles.size(); ++i){
			sampleFace_CD.push_back(CurveDescriptor(sampleFace.cycles[i], sample_sample, 3.0, false));
			sampleFace.sample_cycles.push_back(sampleFace_CD.back().curve);
			prim_is_open.push_back(false);
		}
	}
	return;
}

void mangaShow::read_notable(char *filename){
	FILE *notable_file = fopen(filename, "r");

	std::vector<cv::Point2f> a, b, c;

	unsigned int n;
	fscanf(notable_file, "%u\n", &n);

	a.resize(n), b.resize(n), c.resize(n);
	notable.resize(n);
	for (unsigned int i = 0; i < n; ++i){
		fscanf(notable_file, "%f %f\n", &a[i].x, &a[i].y);
		notable[i] = a[i];
	}
	for (unsigned int i = 0; i < n; ++i)
		fscanf(notable_file, "%f %f\n", &b[i].x, &b[i].y);

	printf("----- warp:\n");
	cv::Mat R = cv::estimateRigidTransform(a, b, true);
	std::cout << R << std::endl;
	cv::Mat H = cv::Mat(3, 3, R.type());
	H.at<double>(0, 0) = R.at<double>(0, 0);
	H.at<double>(0, 1) = R.at<double>(0, 1);
	H.at<double>(0, 2) = R.at<double>(0, 2);

	H.at<double>(1, 0) = R.at<double>(1, 0);
	H.at<double>(1, 1) = R.at<double>(1, 1);
	H.at<double>(1, 2) = R.at<double>(1, 2);

	H.at<double>(2, 0) = 0.0;
	H.at<double>(2, 1) = 0.0;
	H.at<double>(2, 2) = 1.0;

	cv::perspectiveTransform(a, c, H);
	double diff = 0;
	for (unsigned int i = 0; i < c.size(); ++i){
		printf("%lf, %lf\n", c[i].x, c[i].y);
		diff += abs(c[i].x - b[i].x) + abs(c[i].y - b[i].y);
	}
	printf("diff = %lf\n", diff);
	return;
}

void mangaShow::find_seed(){
	seeds.clear();

	for (unsigned int i = 0; i < sampleFace.sample_curves.size(); ++i){
		char ostr[100];
		sprintf(ostr, "ansicon -E [46mProcess:[0m ");
		system(ostr);
		compare_curves_with_primitive(sampleFace.sample_curves[i], i, is_cur[i]);
		printf("%d Seeds: %d\n", i, seeds[i].size());
		draw_curves(false);
		sample_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		draw_sample_face(i, color_chips(i));
		char filename[100];
		sprintf(filename, "results/seeds_%d.png", i);
		cv::imwrite(filename, canvas);
	}
	for (unsigned int i = 0; i < sampleFace.sample_cycles.size(); ++i){
		compare_cycles_with_primitive(sampleFace.sample_cycles[i], i + sampleFace.sample_curves.size());
		char ostr[100];
		sprintf(ostr, "ansicon -E [46mProcess:[0m ");
		system(ostr);
		printf("Find %d Seeds (Cycle): %d\n", i + sampleFace.sample_curves.size(), seeds[i + sampleFace.sample_curves.size()].size());
		draw_curves(false);
		sample_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		draw_sample_face(i, color_chips(i + sampleFace.sample_curves.size()), false);
		char filename[100];
		sprintf(filename, "results/seeds_%d.png", i + sampleFace.sample_curves.size());
		cv::imwrite(filename, canvas);
	}

	return;
}

std::vector<std::vector<unsigned int>> mangaShow::build_relative_table(std::vector<std::vector<mgd>> &all_gd, std::vector<std::vector<unsigned int>> &gd_idx, std::vector<unsigned int> &max_i, unsigned int out_i){
	unsigned int n = sampleFace.sample_curves.size() + sampleFace.sample_cycles.size();

	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	std::vector<std::vector<unsigned int>> relative_table;
	relative_table.resize(n);
	for (unsigned int i = 0; i < n; ++i) relative_table[i].resize(n);

	for (unsigned int i = 0; i < n; ++i){
		//draw_curves(false);
		//CurveDescriptor s_d = CurveDescriptor(seeds[i][max_i[i]], graph_sample, 3.0);
		//for (unsigned int j = 1; j < s_d.curve.size(); ++j){
		//	cv::line(img_show,
		//		cv::Point2d(s_d.curve[j - 1].x * ps, s_d.curve[j - 1].y * -ps) + ds,
		//		cv::Point2d(s_d.curve[j].x * ps, s_d.curve[j].y * -ps) + ds,
		//		color_chips(i));
		//}
		//cv::Point2d notable_pnt = get_notable_pnt(s_d, is_cur[i]);
		//cv::circle(img_show, cv::Point2d(notable_pnt.x * ps, notable_pnt.y * -ps) + ds, 2, black, CV_FILLED);
		
		relative_table[i][i] = max_i[i];

		for (unsigned int j = 0; j < n; ++j){
			if (j == i) continue;
			std::vector<mgd> gd_vec = all_gd[gd_idx[i][j]]; // have sorted
			unsigned int min_i = 0;
			for (unsigned int k = 0; k < gd_vec.size(); ++k){
				if (i < j)
				if (max_i[i] == gd_vec[k].i){
					min_i = gd_vec[k].j;
					break;
				}
				if (i > j)
				if (max_i[i] == gd_vec[k].j){
					min_i = gd_vec[k].i;
					break;
				}
			}
		
		//	s_d = CurveDescriptor(seeds[j][min_i], graph_sample, 3.0);
		//	for (unsigned int k = 1; k < s_d.curve.size(); ++k){
		//		cv::line(img_show,
		//			cv::Point2d(s_d.curve[k - 1].x * ps, s_d.curve[k - 1].y * -ps) + ds,
		//			cv::Point2d(s_d.curve[k].x * ps, s_d.curve[k].y * -ps) + ds,
		//			color_chips(j));
		//	}
		//	cv::Point2d notable_pnt = get_notable_pnt(s_d, is_cur[j]);
		//	cv::circle(img_show, cv::Point2d(notable_pnt.x * ps, notable_pnt.y * -ps) + ds, 2, black, CV_FILLED);
			relative_table[i][j] = min_i;
		}
		
		//char can[100];
		//sprintf(can, "results/prim_%u_%u.png", out_i, i);
		//cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
		//cv::imwrite(can, canvas);
	}

	return relative_table;
}

std::vector<std::unordered_map<unsigned int, unsigned int>> mangaShow::calculate_seed_use_count(std::vector<std::vector<unsigned int>> &relative_table){
	unsigned int n = sampleFace.sample_curves.size() + sampleFace.sample_cycles.size();

	std::vector<std::unordered_map<unsigned int, unsigned int>> seed_use_count;
	seed_use_count.resize(n);
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = 0; j < n; ++j){
			if (seed_use_count[i].find(relative_table[j][i]) == seed_use_count[i].end())
				seed_use_count[i][relative_table[j][i]] = 1;
			else
				seed_use_count[i][relative_table[j][i]] ++;
		}
	}
	return seed_use_count;
}

std::vector<unsigned int> mangaShow::calculate_prim_score(std::vector<std::vector<unsigned int>> &relative_table, std::vector<std::unordered_map<unsigned int, unsigned int>> &seed_use_count){
	unsigned int n = sampleFace.sample_curves.size() + sampleFace.sample_cycles.size();

	std::vector<unsigned int> prim_score;
	prim_score.resize(n);
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = 0; j < n; ++j){
			prim_score[i] += seed_use_count[j][relative_table[i][j]];
		}
		printf("===> %u score: %u\n", i, prim_score[i]);
	}
	return prim_score;
}

unsigned int mangaShow::get_max_count_seed(std::vector<std::unordered_map<unsigned int, unsigned int>> &seed_use_count, unsigned int idx){
	unsigned int max_idx = 0, max_score = 0;
	for (const auto i : seed_use_count[idx]){
		if (i.second > max_score){
			max_idx = i.first;
			max_score = i.second;
		}
	}
	return max_idx;
}

void mangaShow::relative_seed(){
	// init
	unsigned int n = sampleFace.sample_curves.size() + sampleFace.sample_cycles.size();
	gd_idx.resize(n);
	for (unsigned int i = 0; i < n; ++i) gd_idx[i].resize(n);
	geo_score.resize(n);
	all_gd.clear();

	clock_t start_t, end_t;
	for (unsigned int t1 = 0; t1 < n; ++t1){
		for (unsigned int t2 = t1 + 1; t2 < n; ++t2){
			std::vector<double> a_r_a = calculate_relative(sampleFace_CD[t1], sampleFace_CD[t2], is_cur[t1], is_cur[t2]);

			std::vector<mgd> gd_vec;
			// TODO: no seed error
			start_t = clock();
			for (unsigned int i = 0; i < seeds[t1].size(); ++i){
				CurveDescriptor a_d = CurveDescriptor(seeds[t1][i], graph_sample, 3.0);
				if (a_d.is_error()) continue;
				for (unsigned int j = 0; j < seeds[t2].size(); ++j){
					CurveDescriptor b_d = CurveDescriptor(seeds[t2][j], graph_sample, 3.0);
					if (b_d.is_error()) continue;

					std::vector<double> b_r_a = calculate_relative(a_d, b_d, is_cur[t1], is_cur[t2]);

					mgd gd = mgd();
					gd.gd = b_r_a;
					gd.i = i, gd.j = j;
					gd_vec.push_back(gd);
				}
			}
			end_t = clock();
			char ostr[100];
			sprintf(ostr, "ansicon -E [46mProcess:[0m ");
			system(ostr);
			printf("Relative %d <-> %d take %lf\n", t1, t2, (end_t - start_t) / (double)CLOCKS_PER_SEC);

			std::vector<double> n_M, n_m;
			n_M.resize(a_r_a.size()), n_m.resize(a_r_a.size());
			for (unsigned int i = 0; i < a_r_a.size(); ++i)
				n_M[i] = std::numeric_limits<double>::min(), n_m[i] = std::numeric_limits<double>::max();

			for (unsigned int i = 0; i < gd_vec.size(); ++i){
				for (unsigned int j = 0; j < a_r_a.size(); ++j){
					double diff;
					if (j == 3) diff = abs(gd_vec[i].gd[j] - a_r_a[j]) < 1 ? abs(gd_vec[i].gd[j] - a_r_a[j]) : 2 - abs(gd_vec[i].gd[j] - a_r_a[j]);
					else diff = abs(gd_vec[i].gd[j] - a_r_a[j]);
					if (diff > n_M[j]) n_M[j] = diff;
					if (diff < n_m[j]) n_m[j] = diff;
				}
			}
			for (unsigned int i = 0; i < gd_vec.size(); ++i){
				gd_vec[i].diff = 0;
				for (unsigned int j = 0; j < a_r_a.size(); ++j){
					if (n_M[j] - n_m[j] < 0.0000001) gd_vec[i].diff += 0;
					else gd_vec[i].diff += (abs(gd_vec[i].gd[j] - a_r_a[j]) - n_m[j]) / (n_M[j] - n_m[j]);
				}
				if (seeds[t1][gd_vec[i].i] == seeds[t2][gd_vec[i].j]) gd_vec[i].diff = 100000;
			}

			std::sort(gd_vec.begin(), gd_vec.end(), mgd_cmp());

			for (unsigned int i = 0; i < 5 && i < gd_vec.size(); ++i){
				if (geo_score[t1].find(gd_vec[i].i) == geo_score[t1].end())
					geo_score[t1][gd_vec[i].i] = 10 - i * 2;
				else
					geo_score[t1][gd_vec[i].i] += 10 - i * 2;

				if (geo_score[t2].find(gd_vec[i].j) == geo_score[t2].end())
					geo_score[t2][gd_vec[i].j] = 10 - i * 2;
				else
					geo_score[t2][gd_vec[i].j] += 10 - i * 2;
                                                       
				//draw_curves(false);
				//draw_relative_seed(t1, t2, gd_vec[i].i, gd_vec[i].j, lime, red);
				//char filename[100];
				//sprintf(filename, "results/relative_%d_%d_%d.png", t1, t2, i);
				//cv::imwrite(filename, canvas);
			}

			gd_idx[t1][t2] = gd_idx[t2][t1] = all_gd.size();
			all_gd.push_back(gd_vec);
		}
	}
	
	return;
}

void mangaShow::llink_seed(){
	unsigned int n = sampleFace.sample_curves.size() + sampleFace.sample_cycles.size();

	char ostr[100];
	for (unsigned int rk = 0; rk < 30; ++rk){
		for (unsigned int i = 0; i < n; ++i){
			for (unsigned int j = i + 1; j < n; ++j){
				if (rk >= all_gd[gd_idx[i][j]].size()) continue;
				
				mlt lt(n);
				lt.seeds[i] = all_gd[gd_idx[i][j]][rk].i;
				lt.seeds[j] = all_gd[gd_idx[i][j]][rk].j;
				lt.total_rank += rk;
				lt.total_diff += all_gd[gd_idx[i][j]][rk].diff;

				for (unsigned int k = 0; k < n; ++k){
					if (k == i || k == j) continue;
					unsigned int min_k = 0, min_rk = std::numeric_limits<unsigned int>::max();
					double min_diff;
					for (unsigned int l = 0; l < seeds[k].size(); ++l){
						unsigned int a_rk, b_rk;
						for (a_rk = 0; a_rk < all_gd[gd_idx[i][k]].size(); ++a_rk){
							mgd &gd = all_gd[gd_idx[i][k]][a_rk];
							if (i < k && lt.seeds[i] == gd.i && l == gd.j ||
								i > k && lt.seeds[i] == gd.j && l == gd.i) break;
						}
						for (b_rk = 0; b_rk < all_gd[gd_idx[j][k]].size(); ++b_rk){
							mgd &gd = all_gd[gd_idx[j][k]][b_rk];
							if (j < k && lt.seeds[j] == gd.i && l == gd.j ||
								j > k && lt.seeds[j] == gd.j && l == gd.i) break;
						}
						if (a_rk == all_gd[gd_idx[i][k]].size() || b_rk == all_gd[gd_idx[j][k]].size()){
							sprintf(ostr, "ansicon -e [41mDebug:[0m Can't find joint!!");
							system(ostr);
						}
						if (a_rk + b_rk < min_rk){
							min_rk = a_rk + b_rk;
							min_k = l;
							min_diff = all_gd[gd_idx[i][k]][a_rk].diff + all_gd[gd_idx[j][k]][b_rk].diff;
						}
					}
					lt.seeds[k] = min_k;
					lt.total_rank += min_rk;
					lt.total_diff += min_diff;
				}
				for (unsigned int k = 0; k < n; ++k){
					if (k == i || k == j) continue;
					for (unsigned int l = k + 1; l < n; ++l){
						if (l == i || l == j) continue;
						unsigned a_rk;
						for (a_rk = 0; a_rk < all_gd[gd_idx[k][l]].size(); ++a_rk){
							mgd &gd = all_gd[gd_idx[k][l]][a_rk];
							if (lt.seeds[k] == gd.i && lt.seeds[l] == gd.j) break;
						}
						if (a_rk == all_gd[gd_idx[k][l]].size()){
							char ostr[100];
							sprintf(ostr, "ansicon -e [41mDebug:[0m Can't find relative!!");
							system(ostr);
						}
						lt.total_rank += a_rk;
					}
				}
				if (std::find(links.begin(), links.end(), lt) == links.end())
					links.push_back(lt);
			}
		}
	}
	std::sort(links.begin(), links.end(), mlt_cmp());

	return;
}


void mangaShow::link_seed(){
	unsigned int n = sampleFace.sample_curves.size() + sampleFace.sample_cycles.size();

	typedef struct gs{
		unsigned int idx;
		double score;
		gs(unsigned int i, double s){ idx = i; score = s; }
	} gs;
	struct gs_cmp{ bool operator()(gs const &a, gs const &b){ return a.score > b.score; }; };

	std::vector<unsigned int> now_seed, max_prim_seed;
	now_seed.resize(n);
	std::vector<std::vector<gs>> prim_gs;
	prim_gs.resize(n);
	for (unsigned int i = 0; i < n; ++i){
		std::vector<gs> a_gs;
		for (auto s : geo_score[i]){
			a_gs.push_back(gs(s.first, s.second));
		}
		std::sort(a_gs.begin(), a_gs.end(), gs_cmp());
		now_seed[i] = a_gs[0].idx;
		prim_gs[i] = a_gs;
	}
	optimal_seed = now_seed;

	std::vector<std::vector<unsigned int>> relative_table;
	std::vector<std::unordered_map<unsigned int, unsigned int>> seed_use_count;
	std::vector<unsigned int> prim_score;
	std::vector<unsigned int>::iterator max_prim, min_prim;
	unsigned int prev_prim = 0, prim_idx = 0, it_count = 0;
	unsigned int max_prim_score = 0, this_prim_max_score = 0;
	for (unsigned int i = 0; i < 30; ++i){
		relative_table = build_relative_table(all_gd, gd_idx, now_seed, i);
		seed_use_count = calculate_seed_use_count(relative_table);
		prim_score = calculate_prim_score(relative_table, seed_use_count);
		max_prim = std::max_element(prim_score.begin(), prim_score.end());
		min_prim = std::min_element(prim_score.begin(), prim_score.end());
		printf("==> max %u prim score: %u\n", max_prim - prim_score.begin(), *max_prim);
		printf("==> min %u prim score: %u\n", min_prim - prim_score.begin(), *min_prim);
		//unsigned int now_score = *max_prim * *min_prim;
		unsigned int now_score = std::accumulate(prim_score.begin(), prim_score.end(), 0);
		if (now_score > max_prim_score){
			max_prim_score = now_score;
			//optimal_seed = now_seed;
			for (unsigned int j = 0; j < n; ++j) optimal_seed[j] = get_max_count_seed(seed_use_count, j);
			if ((double)std::accumulate(prim_score.begin(), prim_score.end(), 0) / (n * n * n) > 0.75){
				printf("%u / %u = %lf\n", std::accumulate(prim_score.begin(), prim_score.end(), 0), n * n * n, (double)std::accumulate(prim_score.begin(), prim_score.end(), 0) / (n * n * n));
				break;
			}
		}

		if (prev_prim != min_prim - prim_score.begin()){
			prim_idx = 0;
			this_prim_max_score = now_score;
		}
		else{
			if (this_prim_max_score > now_score){
				this_prim_max_score = now_score;
				max_prim_seed = optimal_seed;
			}
		}

		if (prim_idx >= prim_gs[min_prim - prim_score.begin()].size()){
			printf("%u / %u\n", prim_idx, prim_gs[min_prim - prim_score.begin()].size());
			//optimal_seed = max_prim_seed;
			break;
		}
		now_seed[min_prim - prim_score.begin()] = prim_gs[min_prim - prim_score.begin()][prim_idx++].idx;
		prev_prim = min_prim - prim_score.begin();
	}

	double relative_diff = 0;
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = 0; j < n; ++j){
			if (i == j) continue;
			std::vector<mgd> gd_vec = all_gd[gd_idx[i][j]]; // have sorted
			unsigned int min_i = 0;
			for (unsigned int k = 0; k < gd_vec.size(); ++k){
				if (i < j && optimal_seed[i] == gd_vec[k].i && optimal_seed[j] == gd_vec[k].j ||
					i > j && optimal_seed[i] == gd_vec[k].j && optimal_seed[j] == gd_vec[k].i){
					relative_diff += gd_vec[k].diff;
					printf("==> %u <-> %u\n", i, j);
				}
			}
		}
	}
	printf("=> relative diff = %lf\n", relative_diff);
}

void mangaShow::ddraw_matching(){
	unsigned int n = sampleFace.sample_curves.size() + sampleFace.sample_cycles.size();
	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	char str[100];
	for (unsigned int rk = 0; rk < 10 && rk < links.size(); ++rk){
		sprintf(str, "results/notable_%u.txt", rk);
		FILE *out_notable = fopen(str, "w");
		fprintf(out_notable, "%d\n", n);

		draw_curves(false);
		cv::Point2d chin_notable;
		for (unsigned int i = 0; i < n; ++i){
			CurveDescriptor s_d = CurveDescriptor(seeds[i][links[rk].seeds[i]], graph_sample, 3.0);
			for (unsigned int j = 1; j < s_d.curve.size(); ++j){
				cv::line(img_show,
					cv::Point2d(s_d.curve[j - 1].x * ps, s_d.curve[j - 1].y * -ps) + ds,
					cv::Point2d(s_d.curve[j].x * ps, s_d.curve[j].y * -ps) + ds,
					color_chips(i));
			}
			cv::Point2d npt = get_notable_pnt(s_d, is_cur[i]);
			//if (i == 0) chin_notable = npt;
			//if (i == 4 && is_cur[i] == 2){
			//	double min_d = std::numeric_limits<double>::max();
			//	CurveDescriptor s_d = CurveDescriptor(seed_curves[i][links[rk].seeds[i]], graph_sample, 3.0);
			//	for (unsigned int j = 0; j < s_d.curve.size(); ++j){
			//		if (cv::norm(chin_notable - s_d.curve[j]) < min_d){
			//			min_d = cv::norm(chin_notable - s_d.curve[j]);
			//			npt = s_d.curve[j];
			//		}
			//	}
			//}
			cv::circle(img_show, cv::Point2d(npt.x * ps, npt.y * -ps) + ds, 2, black, CV_FILLED);
			fprintf(out_notable, "%lf %lf\n", npt.x, npt.y);
		}
		cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
		sprintf(str, "results/canvas_%u.png", rk);
		cv::imwrite(str, canvas);

		for (unsigned int i = 0; i < sampleFace.sample_curves.size(); ++i){
			cv::Point2d npt = get_notable_pnt(sampleFace.sample_curves[i], is_cur[i]);
			fprintf(out_notable, "%lf %lf\n", npt.x, npt.y);
		}
		for (unsigned int i = 0; i < sampleFace.sample_cycles.size(); ++i){
			cv::Point2d npt = get_midpoint(sampleFace.sample_cycles[i]);
			fprintf(out_notable, "%lf %lf\n", npt.x, npt.y);
		}
		fprintf(out_notable, "total rank: %u\n", links[rk].total_rank);

		fclose(out_notable);
	}

	sample_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < sampleFace.sample_curves.size(); ++i){
		for (unsigned int j = 1; j < sampleFace.sample_curves[i].size(); ++j){
			cv::line(sample_show,
				cv::Point2d(sampleFace.sample_curves[i][j - 1].x * ps, sampleFace.sample_curves[i][j - 1].y * -ps) + ds,
				cv::Point2d(sampleFace.sample_curves[i][j].x * ps, sampleFace.sample_curves[i][j].y * -ps) + ds,
				color_chips(i));
		}
		cv::Point2d npt = get_notable_pnt(sampleFace.sample_curves[i], is_cur[i]);
		cv::circle(sample_show, cv::Point2d(npt.x * ps, npt.y * -ps) + ds, 2, black, CV_FILLED);
	}
	for (unsigned int i = 0; i < sampleFace.sample_cycles.size(); ++i){
		for (unsigned int j = 0; j < sampleFace.sample_cycles[i].size(); ++j){
			cv::Point2d p1 = j ? sampleFace.sample_cycles[i][j - 1] : sampleFace.sample_cycles[i].back();
			cv::Point2d p2 = sampleFace.sample_cycles[i][j];
			cv::line(sample_show,
				cv::Point2d(p1.x * ps, p1.y * -ps) + ds, cv::Point2d(p2.x * ps, p2.y * -ps) + ds,
				color_chips(i + sampleFace.sample_curves.size()));
		}
		cv::Point2d npt = get_midpoint(sampleFace.sample_cycles[i]);
		cv::circle(sample_show, cv::Point2d(npt.x * ps, npt.y * -ps) + ds, 2, black, CV_FILLED);
	}
	cv::resize(sample_show, sample_canvas, cv::Size(sample_show.cols * scale, sample_show.rows * scale));
	cv::imwrite("results/sample_canvas.png", sample_canvas);

	return;
}

void mangaShow::draw_matching(){
	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	FILE *out_notable = fopen("results/notable.txt", "w");
	fprintf(out_notable, "%d\n", sampleFace.sample_curves.size() + sampleFace.sample_cycles.size());

	//FILE *out_curve = fopen("results/curve.txt", "w");

	draw_curves(false);
	for (unsigned int i = 0; i < sampleFace.sample_curves.size() + sampleFace.sample_cycles.size(); ++i){
		CurveDescriptor s_d = CurveDescriptor(seeds[i][optimal_seed[i]], graph_sample, 3.0);
		for (unsigned int j = 1; j < s_d.curve.size(); ++j){
			cv::line(img_show,
				cv::Point2d(s_d.curve[j - 1].x * ps, s_d.curve[j - 1].y * -ps) + ds,
				cv::Point2d(s_d.curve[j].x * ps, s_d.curve[j].y * -ps) + ds,
				color_chips(i));
		}
		cv::Point2d npt = get_notable_pnt(s_d, is_cur[i]);
		cv::circle(img_show, cv::Point2d(npt.x * ps, npt.y * -ps) + ds, 2, black, CV_FILLED);
		fprintf(out_notable, "%lf %lf\n", npt.x, npt.y);
	}
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/canvas.png", canvas);

	sample_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < sampleFace.sample_curves.size(); ++i){
		for (unsigned int j = 1; j < sampleFace.sample_curves[i].size(); ++j){
			cv::line(sample_show,
				cv::Point2d(sampleFace.sample_curves[i][j - 1].x * ps, sampleFace.sample_curves[i][j - 1].y * -ps) + ds,
				cv::Point2d(sampleFace.sample_curves[i][j].x * ps, sampleFace.sample_curves[i][j].y * -ps) + ds,
				color_chips(i));
		}
		cv::Point2d npt = get_notable_pnt(sampleFace.sample_curves[i], is_cur[i]);
		cv::circle(sample_show, cv::Point2d(npt.x * ps, npt.y * -ps) + ds, 2, black, CV_FILLED);
		fprintf(out_notable, "%lf %lf\n", npt.x, npt.y);
	}
	for (unsigned int i = 0; i < sampleFace.sample_cycles.size(); ++i){
		for (unsigned int j = 0; j < sampleFace.sample_cycles[i].size(); ++j){
			cv::Point2d p1 = j ? sampleFace.sample_cycles[i][j - 1] : sampleFace.sample_cycles[i].back();
			cv::Point2d p2 = sampleFace.sample_cycles[i][j];
			cv::line(sample_show,
				cv::Point2d(p1.x * ps, p1.y * -ps) + ds, cv::Point2d(p2.x * ps, p2.y * -ps) + ds,
				color_chips(i + sampleFace.sample_curves.size()));
		}
		cv::Point2d npt = get_midpoint(sampleFace.sample_cycles[i]);
		cv::circle(sample_show, cv::Point2d(npt.x * ps, npt.y * -ps) + ds, 2, black, CV_FILLED);
		fprintf(out_notable, "%lf %lf\n", npt.x, npt.y);
	}
	cv::resize(sample_show, sample_canvas, cv::Size(sample_show.cols * scale, sample_show.rows * scale));
	cv::imwrite("results/sample_canvas.png", sample_canvas);

	fclose(out_notable);

	return;
}

void mangaShow::draw_graph(){
	img_show = img_read.clone();
	
	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	for (const cv::Point2d &p : mangaFace.graph_pnts){
		for (const cv::Point2d &q : mangaFace.graph[p]){
			cv::line(img_show, cv::Point2d(p.x * ps, p.y * -ps) + ds, cv::Point2d(q.x * ps, q.y * -ps) + ds, lime);
		}
	}

	for (const cv::Point2d &p : mangaFace.graph_pnts){
		cv::Scalar draw_color;
		switch (mangaFace.graph[p].size()){
			case 1: draw_color = red; break;
			case 2: draw_color = magenta; break;
			case 3: draw_color = blue; break;
			case 4: draw_color = yellow; break;
			default: draw_color = cyan; break;
		}

		cv::circle(img_show, cv::Point2d(p.x * ps, p.y * -ps) + ds, 1, draw_color, CV_FILLED);
	}

	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/topology.png", canvas);

	char ostr[100];
	sprintf(ostr, "ansicon -e [42mLog:[0m Drawing graph done.");
	system(ostr);
	return;
}

void mangaShow::rng_curves_color(){
	if (curves_color.size() != mangaFace.curves.size()) curves_color.resize(mangaFace.curves.size());
	for (unsigned int i = 0; i < curves_color.size(); ++i)
		curves_color[i] = cv::Scalar(rng.uniform(100, 230), rng.uniform(100, 230), rng.uniform(100, 230));
	return;
}

void mangaShow::set_curves_drawable(int index, bool is_draw){
	if (curves_drawable.size() != mangaFace.sample_curves.size()){ 
		curves_drawable.resize(mangaFace.sample_curves.size());
		for (unsigned int i = 0; i < curves_drawable.size(); ++i) curves_drawable[i] = true;
	}

	if (index < 0 || index >= curves_drawable.size())
		for (unsigned int i = 0; i < curves_drawable.size(); ++i)
			curves_drawable[i] = is_draw;
	else curves_drawable[index] = is_draw;

	return;
}

void mangaShow::draw_curves(bool is_colorful){
	if (curves_color.size() != mangaFace.sample_curves.size()) rng_curves_color();
	if (curves_drawable.size() != mangaFace.sample_curves.size()) set_curves_drawable();

	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);
	img_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
		if (curves_drawable[i]){
			for (unsigned int j = 1; j < mangaFace.sample_curves[i].size(); ++j)
				cv::line(img_show,
				cv::Point2d(mangaFace.sample_curves[i][j - 1].x * ps, mangaFace.sample_curves[i][j - 1].y * -ps) + ds,
				cv::Point2d(mangaFace.sample_curves[i][j].x * ps, mangaFace.sample_curves[i][j].y * -ps) + ds,
				is_colorful ? curves_color[i] : gray);
		}
	}
	for (unsigned int i = 0; i < mangaFace.sample_cycles.size(); ++i){
		for (unsigned int j = 0; j < mangaFace.sample_cycles[i].size(); ++j){
			cv::Point2d p1 = !j ? mangaFace.sample_cycles[i].back() : mangaFace.sample_cycles[i][j - 1];
			cv::Point2d p2 = mangaFace.sample_cycles[i][j];
			cv::line(img_show,
				cv::Point2d(p1.x * ps, p1.y * -ps) + ds, cv::Point2d(p2.x * ps, p2.y * -ps) + ds, 
				is_colorful ? curves_color[i] : gray);
		}
	}

	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/curves.png", canvas);

	char ostr[100];
	sprintf(ostr, "ansicon -e [42mLog:[0m Drawing curves done.");
	system(ostr);
	return;
}

void mangaShow::draw_notable(){
	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	draw_curves(false);
	for (unsigned int i = 0; i < notable.size(); ++i){
		cv::circle(img_show, cv::Point2d(notable[i].x * ps, notable[i].y * -ps) + ds, 2, color_chips(i), CV_FILLED);
	}
	cv::line(img_show,
		cv::Point2d(notable[5].x * ps, notable[5].y * -ps) + ds,
		cv::Point2d(notable[6].x * ps, notable[6].y * -ps) + ds,
		red, 1);

	//cv::Point2d ref_pnt = get_reflect_point(notable[0], notable[5], notable[6]);
	cv::Point2d ref_pnt = notable[0] + (get_midpoint(notable[5], notable[6]) - notable[0]) * 2;
	cv::circle(img_show, cv::Point2d(ref_pnt.x * ps, ref_pnt.y * -ps) + ds, 2, color_chips(7), CV_FILLED);

	cv::Point2d prev_pnt;
	for (double t = 0.0; t <= 1.0; t += 0.1){
		cv::Point2d curr_pnt = (1 - t) * (1 - t) * notable[0] + 2 * t * (1 - t) * notable[3] + t * t * ref_pnt;
		if (t > 0.0)
			cv::line(img_show,
				cv::Point2d(prev_pnt.x * ps, prev_pnt.y * -ps) + ds,
				cv::Point2d(curr_pnt.x * ps, curr_pnt.y * -ps) + ds,
				red, 1);
		prev_pnt = curr_pnt;
	}

	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/canvas.png", canvas);

	return;
}

bool mangaShow::is_read_img(){
	return img_read.dims ? true : false;
}

bool mangaShow::is_read_mangaFace(){
	return mangaFace.graph.size() ? true : false;
}

bool mangaShow::is_read_sampleFace(){
	return sampleFace.graph.size() ? true : false;
}

bool mangaShow::is_read_notable(){
	return notable.size() ? true : false;
}

Bitmap ^mangaShow::mat2Bitmap(cv::Mat img){
	Bitmap ^curImage =
		img.channels() == 1 ?
		gcnew Bitmap(img.cols, img.rows, Imaging::PixelFormat::Format8bppIndexed) :
		gcnew Bitmap(img.cols, img.rows, Imaging::PixelFormat::Format24bppRgb);

	Imaging::BitmapData ^bitmapData = curImage->LockBits(System::Drawing::Rectangle(0, 0, curImage->Width, curImage->Height), Imaging::ImageLockMode::ReadWrite, curImage->PixelFormat);
	unsigned char *p = (unsigned char *)bitmapData->Scan0.ToPointer();
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.step; i++) {
			p[0] = img.data[i + j * img.step];
			++p;
		}
		p += bitmapData->Stride - img.step;
	}
	curImage->UnlockBits(bitmapData);
	return curImage;
}

Bitmap ^mangaShow::get_canvas_Bitmap(){
	cv::imwrite("results/canvas.png", canvas);
	return mat2Bitmap(canvas);
}

Bitmap ^mangaShow::get_sample_canvas_Bitmap(){
	cv::imwrite("results/sample_canvas.png", sample_canvas);
	return mat2Bitmap(sample_canvas);
}

std::vector<bool> mangaShow::get_curves_drawable(){
	if (curves_drawable.size() != mangaFace.graph.size()) set_curves_drawable();
	return curves_drawable;
}

void mangaShow::test(){
}

cv::Scalar mangaShow::color_chips(int i){
	std::vector<cv::Scalar> color = deep_color().color_chips;
	if (i < color.size()) return color[i];
	return cv::Scalar(rng.uniform(50, 230), rng.uniform(50, 230), rng.uniform(50, 230));
}

unsigned int mangaShow::get_notable_index(CurveDescriptor a, int is_c){
	switch (is_c){
	case 0: return max_curvature_index(a.curvature);
	case 1: return douglas_peucker(a.curve, 1)[1];
	case 2: return a.curve.size() >> 1;
	default: return a.curve.size() >> 1;
	}
}

cv::Point2d mangaShow::get_notable_pnt(CurveDescriptor a, int is_c){
	if (is_c == 3) return get_midpoint(a.curve);
	else{
		unsigned int idx = get_notable_index(a, is_c);
		if (idx == 0 || idx == a.curve.size() - 1)
			printf("Warning: notable point is end point.\n");
		return a.curve[idx];
	}
}

int mangaShow::normalize_cross_correlation(std::vector<double> a, std::vector<double> b){
	double max_v = 0;
	unsigned int offset = 0;

	double a_avg = 0;
	for (unsigned int i = 0; i < a.size(); ++i)
		a_avg += a[i];
	a_avg /= a.size();

	for (unsigned int i = 0; i < b.size() - a.size(); ++i){
		double b_avg = 0;
		for (unsigned int j = 0; j < a.size(); ++j)
			b_avg += b[i + j];
		b_avg /= a.size();

		double FT = 0, FF = 0, TT = 0;
		for (unsigned int j = 0; j < a.size(); ++j){
			double F = a[j] - a_avg;
			double T = b[i + j] - b_avg;
			FT += F * T;
			FF += F * F;
			TT += T * T;
		}
		double V = FT / sqrt(FF * TT);
		if (V > max_v){
			max_v = V;
			offset = i;
		}
	}
	if (max_v < 0.8) return -1;
	return offset;
}

std::vector<cv::Point2d> mangaShow::compare_curve(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b, int is_c){
	std::vector<cv::Point2d> segment;

	double a_ratio = curve_length(a) / cv::norm(a[0] - a[a.size() - 1]);

	CurveDescriptor a_d = CurveDescriptor(a);
	if (a_d.is_error()) return std::vector<cv::Point2d>();
	CurveDescriptor b_d = CurveDescriptor(b);
	if (b_d.is_error()) return std::vector<cv::Point2d>();

	cv::Point2d a_notable_pnt = get_notable_pnt(a_d, is_c);
	double a_degree = abc_degree(a_d.curve.front(), a_notable_pnt, a_d.curve.back());

	for (double i = 1; i <= scale_max; i += scale_inter){
		segment.clear();
		if (a_d.curvature.size() / i <= 3) return std::vector<cv::Point2d>();
		b_d.scaling_curvature(i);
		if (b_d.scale_curvature.size() < a_d.curvature.size()) continue;

		int offset = normalize_cross_correlation(a_d.curvature, b_d.scale_curvature);

		if (offset < 0) continue;
		std::vector<cv::Point2d> segment;
		for (unsigned int j = 0; j < a_d.curvature.size() / i; ++j)
			segment.push_back(b_d.curve[j + offset / i]);

		double b_length = curve_length(segment);
		double b_distance = cv::norm(segment.front() - segment.back());
		double b_ratio = b_length / b_distance;
		if ((a_ratio - a_ratio * 0.4 > b_ratio) ||
			(a_ratio + a_ratio * 0.5 < b_ratio)) continue;
		
		CurveDescriptor s_d = CurveDescriptor(segment, 3.0, true);
		if (s_d.is_error()) continue;
		cv::Point2d b_notable_pnt = get_notable_pnt(s_d, is_c);
		double b_degree = abc_degree(s_d.curve.front(), b_notable_pnt, s_d.curve.back());
		if ((a_degree - CV_PI / 8 > b_degree) ||
			(a_degree + CV_PI / 8 < b_degree)) continue;
		
		return segment;
	}
	return std::vector<cv::Point2d>();
}

void mangaShow::compare_curve_add_seed(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b, unsigned int p_i, int is_c){
	std::vector<cv::Point2d> seed;

	std::reverse(a.begin(), a.end());
	seed = compare_curve(a, b, is_c);
	if (seed.size() > 0){
		seeds[p_i].push_back(seed);
		seed_curves[p_i].push_back(b);
	}

	std::reverse(a.begin(), a.end());
	seed = compare_curve(a, b, is_c);
	if (seed.size() > 0){
		seeds[p_i].push_back(seed);
		seed_curves[p_i].push_back(b);
	}

	return;
}

void mangaShow::remove_duplication_seed(unsigned int p_i){
	std::sort(seeds[p_i].begin(), seeds[p_i].end(), point2d_vector_size_cmp);
	std::vector<bool> is_dup;
	is_dup.resize(seeds[p_i].size());
	std::vector<std::vector<cv::Point2d>> tmp_seed, tmp_seed_curve;

	for (unsigned int i = 0; i < seeds[p_i].size(); ++i){
		for (unsigned int j = i + 1; j < seeds[p_i].size(); ++j){
			unsigned int dup_count = 0;
			for (unsigned int k = 0; k < seeds[p_i][j].size(); ++k){
				if (std::find(seeds[p_i][i].begin(), seeds[p_i][i].end(), seeds[p_i][j][k]) != seeds[p_i][i].end())
					dup_count++;
			}
			if (dup_count == seeds[p_i][j].size()) is_dup[j] = true;
		}
	}

	tmp_seed = seeds[p_i], tmp_seed_curve = seed_curves[p_i];
	seeds[p_i].clear(), seed_curves.clear();
	for (unsigned int i = 0; i < tmp_seed.size(); ++i){
		if (!is_dup[i]){
			seeds[p_i].push_back(tmp_seed[i]);
			seed_curves[p_i].push_back(tmp_seed_curve[i]);
		}
	}
	return;
}

// if dup return a dup index (p, q] else return -1, -1
cv::Point have_dup_frag(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b){
	int p = -1, q, dir = 0;
	for (unsigned int i = 0; i < a.size(); ++i){
		if ((q = std::find(b.begin(), b.end(), a[i]) - b.begin()) != b.size()){
			p = i;
			break;
		}
	}

	if (p == -1) return cv::Point(-1, -1);
	cv::Point idx(p, 0);

	if (q != b.size() - 1 && p != a.size() - 1 && b[q + 1] == a[p + 1]) dir = 1;
	if (q != 0 && p != a.size() - 1 && b[q - 1] == a[p + 1]) dir = -1;
	
	if (!dir) return cv::Point(-1, -1);
	else if (dir == 1){
		for (; q < b.size(); ++q, ++p) if (p == a.size() || b[q] != a[p]) break;
		idx.y = p;
	}
	else if (dir == -1){
		for (; q >= 0; --q, ++p) if (p == a.size() || b[q] != a[p]) break;
		idx.y = p;
	}

	if ((idx.y - idx.x) / (double)a.size() > 0.8) return idx;

	return cv::Point(-1, -1);
}

void mangaShow::simplify_seed(unsigned int p_i){
	std::vector<std::vector<cv::Point2d>> tmp_seed, idn_seed;
	std::vector<bool> is_simplify;
	is_simplify.resize(seeds[p_i].size());
	tmp_seed = seeds[p_i], idn_seed.clear();
	
	for (unsigned int i = 0; i < tmp_seed.size(); ++i){
		cv::Point idx(-1, -1);
		for (unsigned int j = 0; j < idn_seed.size(); ++j){
			 idx = have_dup_frag(tmp_seed[i], idn_seed[j]);
			 if (idx != cv::Point(-1, -1)){
				 idn_seed[j] = std::vector<cv::Point2d>(tmp_seed[i].begin() + idx.x, tmp_seed[i].begin() + idx.y);
				 break;
			 }
		}
		if (idx == cv::Point(-1, -1)) idn_seed.push_back(tmp_seed[i]);
	}

	seeds[p_i] = idn_seed;
	return;
}

std::vector<cv::Point2d> mangaShow::join_curve(std::vector<cv::Point2d> a, std::vector<cv::Point2d> b){
	if (a.front() == b.front() || a.front() == b.back()) std::reverse(a.begin(), a.end());
	if (a.back() == b.back()) std::reverse(b.begin(), b.end());
	std::vector<cv::Point2d> c = a;
	c.insert(c.end(), b.begin() + 1, b.end());
	return c;
}

// need link curve function
void mangaShow::compare_curves_with_primitive(std::vector<cv::Point2d> sample_curve, unsigned int p_i, int is_c){
	//if (seeds.size() != sampleFace.sample_curves.size() + sampleFace.sample_cycles.size())
	//	seeds.resize(sampleFace.sample_curves.size() + sampleFace.sample_cycles.size());
	//std::vector<bool> curve_is_visited;
	//curve_is_visited.resize(mangaFace.sample_curves.size()); // all will be zero
	//
	//
	//for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
	//	if (mangaFace.sample_curves[i].size() == 0){
	//		printf("Warning: manga face curve 0 size\n");
	//		continue;
	//	}
	//
	//	for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]].size(); ++j){
	//		unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]][j];
	//		if (a_crv == i || mangaFace.sample_curves[a_crv].size() == 0) continue;
	//
	//		std::vector<cv::Point2d> curve = join_curve(mangaFace.sample_curves[a_crv], mangaFace.sample_curves[i]);
	//
	//		for (unsigned int k = 0; k < mangaFace.pnt_to_curve[mangaFace.sample_curves[i].back()].size(); ++k){
	//			unsigned int b_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i].back()][k];
	//			if (b_crv == i || mangaFace.sample_curves[b_crv].size() == 0) continue;
	//
	//			std::vector<cv::Point2d> curve_e = join_curve(curve, mangaFace.sample_curves[b_crv]);
	//			compare_curve_add_seed(sample_curve, curve_e, p_i, is_c);
	//			curve_is_visited[a_crv] = curve_is_visited[i] = curve_is_visited[b_crv] = true;
	//		}
	//	}
	//}
	//printf("3, ");
	//
	//for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
	//	if (curve_is_visited[i] || mangaFace.sample_curves[i].size() == 0) continue;
	//
	//	for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]].size(); ++j){
	//		unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]][j];
	//		if (a_crv <= i || curve_is_visited[a_crv] || mangaFace.sample_curves[a_crv].size() == 0) continue;
	//
	//		std::vector<cv::Point2d> curve = join_curve(mangaFace.sample_curves[a_crv], mangaFace.sample_curves[i]);
	//		compare_curve_add_seed(sample_curve, curve, p_i, is_c);
	//		curve_is_visited[a_crv] = curve_is_visited[i] = true;
	//	}
	//
	//	for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]].size(); ++j){
	//		unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]][j];
	//		if (a_crv <= i || curve_is_visited[a_crv] || mangaFace.sample_curves[a_crv].size() == 0) continue;
	//
	//		std::vector<cv::Point2d> curve = join_curve(mangaFace.sample_curves[i], mangaFace.sample_curves[a_crv]);
	//		compare_curve_add_seed(sample_curve, curve, p_i, is_c);
	//		curve_is_visited[a_crv] = curve_is_visited[i] = true;
	//	}
	//}
	//printf("2, ");
	//
	//for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
	//	if (curve_is_visited[i] || mangaFace.sample_curves[i].size() == 0) continue;
	//
	//	compare_curve_add_seed(sample_curve, mangaFace.sample_curves[i], p_i, is_c);
	//	curve_is_visited[i] = true;
	//}
	//printf("1. ");
	//
	//remove_duplication_seed(p_i);
	//simplify_seed(p_i);

	if (seeds.size() != sampleFace.sample_curves.size() + sampleFace.sample_cycles.size())
		seeds.resize(sampleFace.sample_curves.size() + sampleFace.sample_cycles.size());
	if (seed_curves.size() != sampleFace.sample_curves.size() + sampleFace.sample_cycles.size())
		seed_curves.resize(sampleFace.sample_curves.size() + sampleFace.sample_cycles.size());
	std::vector<bool> curve_is_visited;
	curve_is_visited.resize(mangaFace.sample_curves.size()); // all will be zero


	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
		if (mangaFace.sample_curves[i].size() == 0) continue;

		std::vector<cv::Point2d> curve;
		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]][j];
			if (a_crv == i || mangaFace.sample_curves[a_crv].size() == 0) continue;
			curve.clear();

			if (mangaFace.sample_curves[a_crv][0] == mangaFace.sample_curves[i][0])
				std::reverse(mangaFace.sample_curves[a_crv].begin(), mangaFace.sample_curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.sample_curves[a_crv].size(); ++k) curve.push_back(mangaFace.sample_curves[a_crv][k]);
			for (unsigned int k = 1; k < mangaFace.sample_curves[i].size(); ++k) curve.push_back(mangaFace.sample_curves[i][k]);

			std::vector<cv::Point2d> curve_e;
			for (unsigned int k = 0; k < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]].size(); ++k){
				unsigned int b_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]][k];
				if (b_crv == i || mangaFace.sample_curves[b_crv].size() == 0) continue;
				curve_e = curve;

				if (mangaFace.sample_curves[b_crv][mangaFace.sample_curves[b_crv].size() - 1] == mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1])
					std::reverse(mangaFace.sample_curves[b_crv].begin(), mangaFace.sample_curves[b_crv].end());
				for (unsigned int l = 1; l < mangaFace.sample_curves[b_crv].size(); ++l) curve_e.push_back(mangaFace.sample_curves[b_crv][l]);

				compare_curve_add_seed(sample_curve, curve_e, p_i, is_c);
				curve_is_visited[a_crv] = curve_is_visited[i] = curve_is_visited[b_crv] = true;
			}
		}
	}
	printf("3, ");

	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
		if (curve_is_visited[i] || mangaFace.sample_curves[i].size() == 0) continue;

		std::vector<cv::Point2d> curve;
		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][0]][j];
			if (a_crv <= i || curve_is_visited[a_crv] || mangaFace.sample_curves[a_crv].size() == 0) continue;
			curve.clear();

			if (mangaFace.sample_curves[a_crv][0] == mangaFace.sample_curves[i][0])
				std::reverse(mangaFace.sample_curves[a_crv].begin(), mangaFace.sample_curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.sample_curves[a_crv].size(); ++k) curve.push_back(mangaFace.sample_curves[a_crv][k]);
			for (unsigned int k = 1; k < mangaFace.sample_curves[i].size(); ++k) curve.push_back(mangaFace.sample_curves[i][k]);

			compare_curve_add_seed(sample_curve, curve, p_i, is_c);
			curve_is_visited[a_crv] = curve_is_visited[i] = true;
		}

		for (unsigned int j = 0; j < mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]].size(); ++j){
			unsigned int a_crv = mangaFace.pnt_to_curve[mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1]][j];
			if (a_crv <= i || curve_is_visited[a_crv] || mangaFace.sample_curves[a_crv].size() == 0) continue;
			curve.clear();

			if (mangaFace.sample_curves[a_crv][mangaFace.sample_curves[a_crv].size() - 1] == mangaFace.sample_curves[i][mangaFace.sample_curves[i].size() - 1])
				std::reverse(mangaFace.sample_curves[a_crv].begin(), mangaFace.sample_curves[a_crv].end());
			for (unsigned int k = 0; k < mangaFace.sample_curves[i].size(); ++k) curve.push_back(mangaFace.sample_curves[i][k]);
			for (unsigned int k = 1; k < mangaFace.sample_curves[a_crv].size(); ++k) curve.push_back(mangaFace.sample_curves[a_crv][k]);

			compare_curve_add_seed(sample_curve, curve, p_i, is_c);
			curve_is_visited[a_crv] = curve_is_visited[i] = true;
		}
	}
	printf("2, ");

	for (unsigned int i = 0; i < mangaFace.sample_curves.size(); ++i){
		if (curve_is_visited[i] || mangaFace.sample_curves[i].size() == 0) continue;

		compare_curve_add_seed(sample_curve, mangaFace.sample_curves[i], p_i, is_c);
		curve_is_visited[i] = true;
	}
	printf("1. ");

	remove_duplication_seed(p_i);
	simplify_seed(p_i);
	return;
}

void mangaShow::compare_cycles_with_primitive(std::vector<cv::Point2d> sample_cycle, unsigned int p_i){
	if (seeds.size() != sampleFace.sample_curves.size() + sampleFace.sample_cycles.size())
		seeds.resize(sampleFace.sample_curves.size() + sampleFace.sample_cycles.size());

	cv::Point2d a_mass = get_midpoint(sample_cycle);
	std::vector<cv::Point2f> sample_cycle_float;
	for (const cv::Point2d p : sample_cycle) sample_cycle_float.push_back(cv::Point2f(p));
	double a_area = cv::contourArea(sample_cycle_float);
	double a_perimeter = curve_length(sample_cycle, false);
	double a_max_r = 0;
	for (const cv::Point2d p : sample_cycle) a_max_r = std::max(a_max_r, cv::norm(p - a_mass));

	std::vector<cv::Point2d> cycle;
	for (unsigned int i = 0; i < mangaFace.sample_cycles.size(); ++i){
		cycle = mangaFace.sample_cycles[i];
		cv::Point2d b_mass = get_midpoint(cycle);
		std::vector<cv::Point2f> cycle_float;
		for (const cv::Point2d p : cycle) cycle_float.push_back(cv::Point2f(p));
		double b_area = cv::contourArea(cycle_float);
		double b_perimeter = curve_length(cycle, false);
		double b_max_r = 0;
		for (const cv::Point2d p : cycle) b_max_r = std::max(b_max_r, cv::norm(p - b_mass));
		if (abs(b_max_r / a_max_r - 1) < max_r_thresh) seeds[p_i].push_back(cycle);
	}

	for (unsigned int i = 0; i < mangaFace.junction_cycles.size(); ++i){
		std::vector<unsigned int> cycle_idx = mangaFace.junction_cycles[i];
		cycle = mangaFace.sample_curves[cycle_idx[0]];
		if(cycle_idx.size() > 1){
			std::vector<cv::Point2d> cycle_e1 = mangaFace.sample_curves[cycle_idx[1]];
			if (cycle[0] == cycle_e1[0] || cycle[0] == cycle_e1.back())
				std::reverse(cycle.begin(), cycle.end());
			for (unsigned int j = 1; j < cycle_idx.size(); ++j){
				std::vector<cv::Point2d> cycle_ei = mangaFace.sample_curves[cycle_idx[j]];
				if (cycle_ei.back() == cycle.back())
					std::reverse(cycle_ei.begin(), cycle_ei.end());
				for (unsigned int k = 1; k < cycle_ei.size(); ++k)
					cycle.push_back(cycle_ei[k]);
			}
		}
		cycle.pop_back(); // remove back which same with first

		cv::Point2d b_mass = get_midpoint(cycle);
		std::vector<cv::Point2f> cycle_float;
		for (const cv::Point2d p : cycle) cycle_float.push_back(cv::Point2f(p));
		double b_area = cv::contourArea(cycle_float);
		double b_perimeter = curve_length(cycle, false);
		double b_max_r = 0;
		for (const cv::Point2d p : cycle) b_max_r = std::max(b_max_r, cv::norm(p - b_mass));
		if (abs(b_max_r / a_max_r - 1) < max_r_thresh) seeds[p_i].push_back(cycle);
	}
	return;
}

std::vector<double> mangaShow::calculate_relative(CurveDescriptor a, CurveDescriptor b, int a_c, int b_c){
	std::vector<double> relative;
	relative.resize(5);

	cv::Point2d a_npt = get_notable_pnt(a, a_c);
	cv::Point2d b_npt = get_notable_pnt(b, b_c);

	cv::Point2d t1, t2;
	if(a_c != 3) t1 = caculate_tangent(a.curve, get_notable_index(a, a_c));
	if(b_c != 3) t2 = caculate_tangent(b.curve, get_notable_index(b, b_c));
	cv::Point2d n1 = b_npt - a_npt;
	cv::Point2d n2 = -n1;

	// relative_angle
	if (a_c == 3) relative[0] = 0.5;
	else relative[0] = (n1.cross(t1) >= 0 ? v_degree(n1, t1) : v_degree(n1, -t1)) / CV_PI;
	if (b_c == 3) relative[1] = 0.5;
	else relative[1] = (n2.cross(t2) >= 0 ? v_degree(n2, t2) : v_degree(n2, -t2)) / CV_PI;

	cv::Point2d c = get_midpoint(a_npt, b_npt);
	double r = cv::norm(a_npt - c);
	std::vector<cv::Point2d> pnts_in_circle;
	for (unsigned int i = 0; i < a.curve.size(); ++i)
		if (cv::norm(a.curve[i] - c) < r) pnts_in_circle.push_back(a.curve[i]);
	for (unsigned int i = 0; i < b.curve.size(); ++i)
		if (cv::norm(b.curve[i] - c) < r) pnts_in_circle.push_back(b.curve[i]);
	cv::Point2d m = get_midpoint(pnts_in_circle);

	// center_mass_distance
	relative[2] = cv::norm(m - c) / r * ((m - c).cross(a_npt - c) > 0 ? 1 : -1);
	// center_mass_angle
	relative[3] = v_angle(a_npt - c, m - c) / CV_PI;
	// distance
	relative[4] = cv::norm(a_npt - b_npt);

	return relative;
}

template<typename T>
T &mangaShow::ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c){
	return ((T *)m.data)[(p.y * m.cols + p.x) * m.channels() + c];
}

double mangaShow::curve_length(std::vector<cv::Point2d> curve, bool is_open){
	double total = 0;
	if (!is_open) total += cv::norm(curve.back() - curve[0]);
	for (unsigned int i = 1; i < curve.size(); ++i)	total += cv::norm(curve[i] - curve[i - 1]);
	return total;
}

cv::Point2d mangaShow::get_midpoint(cv::Point2d a, cv::Point2d b){
	cv::Point2d m = a + b;
	return cv::Point2d(m.x / 2, m.y / 2);
}

cv::Point2d mangaShow::get_midpoint(std::vector<cv::Point2d> pnts){
	cv::Point2d m = cv::Point2d(0, 0);
	if (pnts.size() == 0) return m;
	for (unsigned int i = 0; i < pnts.size(); ++i) m += pnts[i];
	return cv::Point2d(m.x / pnts.size(), m.y / pnts.size());
}

// mx + y = c
cv::Point2d mangaShow::get_reflect_point(cv::Point2d a, cv::Point2d p, cv::Point2d q){
	cv::Point2d n = q - p;
	double m = -n.y / n.x;
	double c = m * p.x + p.y;
	double t = (a.x * n.x + (a.y - c) * n.y) / (n.x - m * n.y);
	cv::Point2d b(t, c - m * t);
	return a + (b - a) * 2;
}

// p to p1<->p2 line distance
double mangaShow::perpendicular_distance(cv::Point2d p, cv::Point2d p1, cv::Point2d p2){
	cv::Point2d v1 = p - p1, v2 = p2 - p1;
	return std::fabs(v1.cross(v2)) / cv::norm(p1 - p2);
}

// only need input line & max_depth; p, q, depth using on recursive;
std::vector<unsigned int> mangaShow::douglas_peucker(std::vector<cv::Point2d> &line, int max_depth, int p, int q, int depth){
	if (q == -1) q = line.size();
	if (p + 1 >= q - 1 || depth >= max_depth) return std::vector<unsigned int>();

	double max_dis = 0;
	int max_index = -1;
	std::vector<unsigned int> angle_index, sub_angle_index;
	for (size_t i = p + 1; i < q - 1; ++i){
		double dis = perpendicular_distance(line[i], line[p], line[q - 1]);
		if (dis > max_dis){
			max_dis = dis;
			max_index = i;
		}
	}
	if (max_index == -1) return std::vector<unsigned int>();
	angle_index.push_back(max_index);
	sub_angle_index = douglas_peucker(line, max_depth, p, (p + q) >> 1, depth + 1);
	for (size_t i = 0; i < sub_angle_index.size(); ++i) angle_index.push_back(sub_angle_index[i]);
	sub_angle_index = douglas_peucker(line, max_depth, (p + q) >> 1, q, depth + 1);
	for (size_t i = 0; i < sub_angle_index.size(); ++i) angle_index.push_back(sub_angle_index[i]);

	if (p == 0 && q == line.size()){ // ³Ìªí¼h±NºÝÂI¥[¤J
		angle_index.push_back(p);
		angle_index.push_back(q - 1);
	}
	// ¥h±¼­«½ÆªºÂI
	std::sort(angle_index.begin(), angle_index.end());
	std::vector<unsigned int>::iterator it = std::unique(angle_index.begin(), angle_index.end());
	angle_index.resize(std::distance(angle_index.begin(), it));

	return angle_index;
}

unsigned int mangaShow::max_curvature_index(std::vector<double> curvature){
	double max_curvature = 0;
	unsigned int max_index = 0;
	for (unsigned int i = 0; i < curvature.size(); ++i){
		if (curvature[i] > max_curvature){
			max_curvature = curvature[i];
			max_index = i;
		}
	}
	return max_index;
}

// return [0, PI]
double mangaShow::v_degree(cv::Point2d v1, cv::Point2d v2){
	return acos(v1.ddot(v2) / cv::norm(v1) / cv::norm(v2));
}

// v2 - v1 angle
double mangaShow::v_angle(cv::Point2d v1, cv::Point2d v2){
	double a1 = atan2(v1.y, v1.x);
	double a2 = atan2(v2.y, v2.x);
	return a2 - a1;
}

// return [0, PI]
double mangaShow::abc_degree(cv::Point2d a, cv::Point2d b, cv::Point2d c){
	cv::Point2d v1 = a - b, v2 = c - b;
	return v_degree(v1, v2);
}

cv::Point2d mangaShow::caculate_tangent(std::vector<cv::Point2d> curve, unsigned int index){
	if (curve.size() < 2){
		printf("... relative curve small than 2\n");
		return cv::Point2d(0, 0);
	}

	if (index == 0)	return curve[index + 1] - curve[index];
	if (index == curve.size() - 1) return curve[index] - curve[index - 1];
	if (index == 1 || index == curve.size() - 2){
		cv::Point2d v1 = curve[index] - curve[index - 1];
		cv::Point2d v2 = curve[index + 1] - curve[index];
		cv::Point2d t = v1 + v2;
		return cv::Point2d(t.x / 2, t.y / 2);

	}
	cv::Point2d v0 = curve[index - 1] - curve[index - 2];
	cv::Point2d v1 = curve[index] - curve[index - 1];
	cv::Point2d v2 = curve[index + 1] - curve[index];
	cv::Point2d v3 = curve[index + 2] - curve[index + 1];
	cv::Point2d t = v0 * 1 + v1 * 3 + v2 * 3 + v3 * 1;
	return cv::Point2d(t.x / 8, t.y / 8);
}

void mangaShow::draw_sample_face(int sample, cv::Scalar color, bool is_open){
	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	if (sample == -1){
		sample_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		for (unsigned int i = 0; i < sampleFace.sample_curves.size(); ++i){
			for (unsigned int j = 1; j < sampleFace.sample_curves[i].size(); ++j)
				cv::line(sample_show,
				cv::Point2d(sampleFace.sample_curves[i][j - 1].x * ps, sampleFace.sample_curves[i][j - 1].y * -ps) + ds,
				cv::Point2d(sampleFace.sample_curves[i][j].x * ps, sampleFace.sample_curves[i][j].y * -ps) + ds,
				color_chips(i));
		}
		for (unsigned int i = 0; i < sampleFace.sample_cycles.size(); ++i){
			for (unsigned int j = 0; j < sampleFace.sample_cycles[i].size(); ++j){
				cv::Point2d p1 = j ? sampleFace.sample_cycles[i][j - 1] : sampleFace.sample_cycles[i].back();
				cv::Point2d p2 = sampleFace.sample_cycles[i][j];
				cv::line(sample_show,
					cv::Point2d(p1.x * ps, p1.y * -ps) + ds, cv::Point2d(p2.x * ps, p2.y * -ps) + ds,
					color_chips(i + sampleFace.sample_curves.size()));
			}
		}
		cv::resize(sample_show, sample_canvas, cv::Size(sample_show.cols * scale, sample_show.rows * scale));
		return;
	}

	std::vector<cv::Point2d> &draw_prim = is_open ? sampleFace.sample_curves[sample] : sampleFace.sample_cycles[sample];
	for (unsigned int i = is_open ? 1 : 0; i < draw_prim.size(); ++i){
		cv::Point2d p1 = i ? draw_prim[i - 1] : draw_prim.back();
		cv::Point2d p2 = draw_prim[i];
		cv::line(sample_show,
			cv::Point2d(p1.x * ps, p1.y * -ps) + ds, cv::Point2d(p2.x * ps, p2.y * -ps) + ds,
			color);
	}
	cv::resize(sample_show, sample_canvas, cv::Size(sample_show.cols * scale, sample_show.rows * scale));

	std::vector<std::vector<cv::Point2d>> &draw_seeds = is_open ? seeds[sample] : seeds[sample + sampleFace.sample_curves.size()];
	for (unsigned int i = 0; i < draw_seeds.size(); ++i){
		for (unsigned int j = is_open ? 1 : 0; j < draw_seeds[i].size(); ++j){
			cv::Point2d p1 = j ? draw_seeds[i][j - 1] : draw_seeds[i].back();
			cv::Point2d p2 = draw_seeds[i][j];
			cv::line(img_show,
				cv::Point2d(p1.x * ps, p1.y * -ps) + ds, cv::Point2d(p2.x * ps, p2.y * -ps) + ds,
				color);
		}
	}
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));

	return;
}

void mangaShow::draw_relative_seed(unsigned int p_i, unsigned int p_j, unsigned int s_i, unsigned int s_j, cv::Scalar s_i_c, cv::Scalar s_j_c){
	int ps = std::max(img_read.rows, img_read.cols);
	cv::Point2d ds(img_read.cols >> 1, img_read.rows >> 1);

	for (unsigned int i = 1; i < seeds[p_i][s_i].size(); ++i){
		cv::line(img_show,
			cv::Point2d(seeds[p_i][s_i][i - 1].x * ps, seeds[p_i][s_i][i - 1].y * -ps) + ds,
			cv::Point2d(seeds[p_i][s_i][i].x * ps, seeds[p_i][s_i][i].y * -ps) + ds,
			s_i_c);
	}
	for (unsigned int i = 1; i < seeds[p_j][s_j].size(); ++i){
		cv::line(img_show,
			cv::Point2d(seeds[p_j][s_j][i - 1].x * ps, seeds[p_j][s_j][i - 1].y * -ps) + ds,
			cv::Point2d(seeds[p_j][s_j][i].x * ps, seeds[p_j][s_j][i].y * -ps) + ds,
			s_j_c);
	}
	cv::resize(img_show, canvas, cv::Size(img_show.cols * scale, img_show.rows * scale));
}

void mangaShow::draw_plot_graph(std::vector<cv::Point2d> data, char *win_name){
	int width = 1200, height = 900, padding = 50;
	cv::Mat plot_show = cv::Mat(height, width, CV_8UC3);
	std::vector<cv::Point2d>::iterator max_x = std::max_element(data.begin(), data.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator min_x = std::min_element(data.begin(), data.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator max_y = std::max_element(data.begin(), data.end(), p_y_cmp);
	std::vector<cv::Point2d>::iterator min_y = std::min_element(data.begin(), data.end(), p_y_cmp);

	double norm_x = (*max_x).x - (*min_x).x, norm_y = (*max_y).y - (*min_y).y;
	int inner_w = width - padding * 2, inner_h = height - padding * 2;
	for (unsigned int i = 1; i < data.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data[i - 1].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data[i].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data[i].y - (*min_y).y) / norm_y * inner_h)),
			cyan,
			2);
	}
	printf("max_y: %lf, min_y: %lf\n", (*max_y).y, (*min_y).y);
	cv::imshow(win_name, plot_show);
	cv::imwrite("results/plot.png", plot_show);
}

void mangaShow::draw_plot_graph(std::vector<cv::Point2d> data_a, std::vector<cv::Point2d> data_b, double offset, char *win_name){
	int width = 1200, height = 900, padding = 50;
	cv::Mat plot_show = cv::Mat(height, width, CV_8UC3);
	std::vector<cv::Point2d>::iterator max_x = std::max_element(data_b.begin(), data_b.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator min_x = std::min_element(data_b.begin(), data_b.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator max_y = std::max_element(data_b.begin(), data_b.end(), p_y_cmp);
	std::vector<cv::Point2d>::iterator min_y = std::min_element(data_b.begin(), data_b.end(), p_y_cmp);

	double norm_x = (*max_x).x - (*min_x).x, norm_y = (*max_y).y - (*min_y).y;
	int inner_w = width - padding * 2, inner_h = height - padding * 2;
	for (unsigned int i = 1; i < data_b.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data_b[i - 1].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_b[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data_b[i].x - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_b[i].y - (*min_y).y) / norm_y * inner_h)),
			cyan,
			2);
	}
	for (unsigned int i = 1; i < data_a.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data_a[i - 1].x + offset - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_a[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data_a[i].x + offset - (*min_x).x) / norm_x * inner_w), padding + inner_h - round((data_a[i].y - (*min_y).y) / norm_y * inner_h)),
			yellow,
			2);
	}
	//printf("max_y: %lf, min_y: %lf\n", (*max_y).y, (*min_y).y);
	printf("=> Drawing %s plot done.\n", win_name);
	cv::imshow(win_name, plot_show);
	cv::imwrite("results/plot.png", plot_show);
}