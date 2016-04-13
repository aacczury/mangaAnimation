#include "mangaShow.h"


mangaShow::mangaShow(char *filename){
	img_read = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	// need exception handling
	img_show = img_read.clone();
	cv::resize(img_show, img_show_scale, cv::Size(img_show.cols * scale, img_show.rows * scale));
}

void mangaShow::vector_curves(){
	cv::Mat img_gray, img_gray_32f;
	cv::cvtColor(img_read, img_gray, CV_BGR2GRAY);
	img_gray.convertTo(img_gray_32f, CV_32FC1, 1.0 / 255);

	VectorCurve vc(img_gray_32f);
	vc.CalSecDer();
	curves = vc.Link();

	set_curves_drawable();
	printf("=> Finding %d curve done.\n", curves.size());
	return;
}


bool cmp_curve_size(std::vector<cv::Point> &a, std::vector<cv::Point> &b){
	return a.size() > b.size();
}
void mangaShow::remove_dump_by_ROI(unsigned short thickness){
	std::sort(curves.begin(), curves.end(), cmp_curve_size);

	for (unsigned int i = 0; i < curves.size(); ++i){
		if (!curves_drawable[i]) continue;

		cv::Mat ROI = cv::Mat(img_read.rows, img_read.cols, CV_8UC1, cv::Scalar(0));
		for (unsigned int j = 1; j < curves[i].size(); ++j)
			cv::line(ROI, curves[i][j - 1], curves[i][j], cv::Scalar(1), thickness);

		for (unsigned int j = i + 1; j < curves.size(); ++j){
			if (!curves_drawable[j]) continue;

			unsigned int useless_pnt = 0;
			for (unsigned int k = 0; k < curves[j].size(); ++k){
				if (ref_Mat_val(ROI, uchar(0), curves[j][k]))
					useless_pnt++;
			}

			if (useless_pnt == curves[j].size())
				set_curves_drawable(j, false);
		}
	}
	return;
}

void mangaShow::topol_curves(){
	for (unsigned int i = 0; i < curves.size(); ++i){
		if (!curves_drawable[i]) continue;

		curves_pnts.insert(curves[i][0]);
		for (unsigned int j = 1; j < curves[i].size(); ++j){
			curves_pnts.insert(curves[i][j]);
			topol[curves[i][j - 1]].insert(curves[i][j]);
			topol[curves[i][j]].insert(curves[i][j - 1]);
		}
	}
	return;
}

void mangaShow::link_adjacent(){ // maybe need speed up
	for (std::unordered_set<cv::Point>::iterator pit = curves_pnts.begin(); pit != curves_pnts.end(); ++pit){
		std::unordered_set<cv::Point>::iterator qit = pit;
		qit++;
		for (; qit != curves_pnts.end(); ++qit){
			cv::Point dis = *pit - *qit;
			dis = cv::Point(abs(dis.x), abs(dis.y));
			if ((dis.x + dis.y) > 0 && (dis.x + dis.y) <= 2 && dis.x <= 1 && dis.y <= 1){
				topol[*pit].insert(*qit);
				topol[*qit].insert(*pit);
			}
		}
	}
	return;
}

void mangaShow::caculate_curve(){
	std::vector<cv::Point2d> curve;
	for (unsigned int i = 0; i < curves[0].size(); ++i)
		curve.push_back(curves[0][i]);

	CurveDescriptor cd = CurveDescriptor(curve, curve.size() / 5, 3.0, true);
	curve = cd.get_smooth_curve();
	std::vector<double> curvature = cd.get_curvature();
	std::vector<double> curvature_integration = cd.get_curvature_integration();

	//double sigma = 3.0;
	//int M = round((10.0 * sigma + 1.0) / 2.0) * 2 - 1;
	//assert(M % 2 == 1); //M is an odd number
	//
	//vector<double> curve_x, curve_y;
	//PolyLineSplit(curve, curve_x, curve_y);
	//vector<double> sample_x, sample_y;
	//ResampleCurve(curve_x, curve_y, sample_x, sample_y, curve.size() / 5, true);
	//std::vector<double> g, dg, ddg;
	//vector<double> curvature;
	//vector<double> smooth_x, smooth_y;
	//ComputeCurveCSS(sample_x, sample_y, curvature, smooth_x, smooth_y, sigma, true);
	//PolyLineMerge(curve, smooth_x, smooth_y);

	for (unsigned int i = 1; i < curve.size(); ++i){
		cv::line(img_show, cv::Point(round(curve[i].x), round(curve[i].y)), cv::Point(round(curve[i - 1].x), round(curve[i - 1].y)), cv::Scalar(0, 0, 255));
	}
	cv::resize(img_show, img_show_scale, cv::Size(img_show.cols * 3, img_show.rows * 3));

	vector<cv::Point2d> data;
	data.resize(curve.size());
	for (unsigned int i = 0; i < curve.size() - 1; ++i){
		if (i == 0) data[i] = cv::Point2d(0, abs(curvature_integration[i]));
		else data[i] = cv::Point2d(data[i - 1].x + cv::norm(curve[i] - curve[i - 1]), abs(curvature_integration[i]));
		printf("%lf\n", curvature_integration[i]);
	}
	draw_plot_graph(data);
}

void mangaShow::rng_curves_color(){
	if (curves_color.size() != curves.size())
		curves_color.resize(curves.size());
	for (unsigned int i = 0; i < curves_color.size(); ++i)
		curves_color[i] = cv::Scalar(rng.uniform(0, 221), rng.uniform(0, 221), rng.uniform(0, 221));
	return;
}

void mangaShow::set_curves_drawable(int index, bool is_draw){
	if (curves_drawable.size() != curves.size()){
		curves_drawable.resize(curves.size());
		for (unsigned int i = 0; i < curves_drawable.size(); ++i)
			curves_drawable[i] = true;
	}

	if (index < 0 || index >= curves_drawable.size())
		for (unsigned int i = 0; i < curves_drawable.size(); ++i)
			curves_drawable[i] = is_draw;
	else curves_drawable[index] = is_draw;

	return;
}

void mangaShow::draw_curves(){
	if (curves_color.size() != curves.size()) rng_curves_color();
	if (curves_drawable.size() != curves.size()) set_curves_drawable();

	img_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (unsigned int i = 0; i < curves.size(); ++i)
		for (unsigned int j = 1; j < curves[i].size(); ++j)
			if (curves_drawable[i])
				cv::line(img_show, curves[i][j - 1], curves[i][j], curves_color[i]);
	cv::resize(img_show, img_show_scale, cv::Size(img_show.cols * scale, img_show.rows * scale));
	return;
}

void mangaShow::draw_topol(){
	img_show = cv::Mat(img_read.rows, img_read.cols, CV_8UC3, cv::Scalar(255, 255, 255));

	typedef struct p_degree{
		p_degree(cv::Point _p, unsigned int _d) :p(_p), degree(_d){};
		cv::Point p;
		unsigned int degree;
	} p_degree;
	
	struct p_degree_cmp{
		bool operator()(p_degree const &a, p_degree const &b){
			return a.degree < b.degree;
		}
	};

	std::vector<p_degree> pnts_degree;
	for (const cv::Point &p : curves_pnts)
		pnts_degree.push_back(p_degree(p, topol[p].size()));
	std::sort(pnts_degree.begin(), pnts_degree.end(), p_degree_cmp());

	for (const p_degree &pnt_degree : pnts_degree){
		cv::Point p = pnt_degree.p;
		//printf("size: %d\n", pnt_degree.degree);
		cv::Scalar draw_color;
		switch (topol[p].size()){
			case 1: case 2: draw_color = gray; break;
			case 3: draw_color = blue; break;
			case 4: draw_color = cyan; break;
			case 5: draw_color = green; break;
			case 6: draw_color = yellow; break;
			default: draw_color = red; break;
		}

		for (const cv::Point &q : topol[p])
			cv::line(img_show, p, q, draw_color);
	}
	cv::resize(img_show, img_show_scale, cv::Size(img_show.cols * scale, img_show.rows * scale));
	cv::imwrite("results/topology.png", img_show_scale);
	return;
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
	return mat2Bitmap(img_show_scale);
}

std::vector<bool> mangaShow::get_curves_drawable(){
	if (curves_drawable.size() != curves.size()) set_curves_drawable();
	return curves_drawable;
}

void mangaShow::test(){

}


template<typename T>
// type:Mat type ex: uchar(0), i: row, j: col, c: channel
T &mangaShow::ref_Mat_val(cv::Mat &m, T type, int i, int j, int c){
	return ((T *)m.data)[(i * m.cols + j) * m.channels() + c];
}

template<typename T>
// type:Mat type ex: uchar(0), i: row, j: col, c: channel
T &mangaShow::ref_Mat_val(cv::Mat &m, T type, cv::Point p, int c){
	return ((T *)m.data)[(p.y * m.cols + p.x) * m.channels() + c];
}

void mangaShow::integral_curvature_curve(std::vector<cv::Point2d> curve){
	// O O O O ... O O O O curve
	// X X O O ... O O X X curvature
	// X X X O ... O X X X curvature_integration
	std::vector<double> curvature;
	std::vector<double> curvature_integration;
	curvature.resize(curve.size());
	for (unsigned int i = 6; i < curve.size() - 6; ++i){
		//double dx1 = curve[i].x - curve[i - 1].x;
		//double dx2 = curve[i + 1].x - curve[i].x;
		//double dy1 = curve[i].y - curve[i - 1].y;
		//double dy2 = curve[i + 1].y - curve[i].y;
		//double ddx = dx2 - dx1;
		//double ddy = dy2 - dy1;
		//
		//if (pow(dx1 * dx1 + dy1 * dy1, 1.5) == 0) curvature.push_back(0);
		//else curvature.push_back((dx1 * ddy - dy1 * ddx) / pow(dx1 * dx1 + dy1 * dy1, 1.5));

		//double a = cv::norm(curve[i] - curve[i - 2]);
		//double b = cv::norm(curve[i + 2] - curve[i]);
		//double c = cv::norm(curve[i + 2] - curve[i - 2]);
		//double R = circumscribed_circle_radius(a, b, c);
		//if(R == 0) curvature.push_back(0);
		//else curvature.push_back(1 / R);
		
		cv::Point2d p1 = curve[i - 6];
		cv::Point2d p2 = curve[i - 3];
		cv::Point2d p3 = curve[i - 0];
		cv::Point2d p4 = curve[i + 3];
		cv::Point2d p5 = curve[i + 6];

		double b1 = (p5.x + p1.x + 2 * p4.x + 2 * p2.x - 6 * p3.x) / 12;
		double b2 = (p5.y + p1.y + 2 * p4.y + 2 * p2.y - 6 * p3.y) / 12;
		double c1 = (p5.x - p1.x + 4 * p4.x + 4 * p2.x) / 12;
		double c2 = (p5.y - p1.y + 4 * p4.y + 4 * p2.y) / 12;
		if (abs(pow(c1 * c1 + c2 * c2, 1.5)) < 0.000001) curvature[i] = 0;
		else curvature[i] = abs(2 * (c1 * b2 - c2 * b1) / pow(c1 * c1 + c2 * c2, 1.5)) * 100;

		printf("curvature: %lf\n", curvature[i]);


		//double ds = cv::norm(curve[i] - curve[i - 1]);
		//curvature_integration.push_back((curvature[i] + curvature[i - 1]) * ds / 2);
		//if (i == 2 || i == curve.size() - 3){
		//	curvature_integration.push_back(curvature_integration[curvature_integration.size() - 1]);
		//	curvature_integration.push_back(curvature_integration[curvature_integration.size() - 1]);
		//}
	}

	std::vector<cv::Point2d> data;
	for (unsigned int i = 0; i < curve.size(); ++i){
		if (!i) data.push_back(cv::Point2d(0, curvature[i]));
		else data.push_back(cv::Point2d(data[data.size() - 1].x + cv::norm(curve[i] - curve[i - 1]), curvature[i]));
	}
	draw_plot_graph(data);
}

unsigned int mangaShow::normalize_cross_correlation(std::vector<double> a, std::vector<double> b){
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

	return offset;
}

bool p_x_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.x < b.x; }
bool p_y_cmp(cv::Point2d const &a, cv::Point2d const &b){ return a.y < b.y; }
void mangaShow::draw_plot_graph(std::vector<cv::Point2d> data){
	int width = 1200, height = 900, padding = 50;
	cv::Mat plot_show = cv::Mat(height, width, CV_8UC3);
	std::sort(data.begin(), data.end(), p_x_cmp);
	std::vector<cv::Point2d>::iterator max_y = std::max_element(data.begin(), data.end(), p_y_cmp);
	std::vector<cv::Point2d>::iterator min_y = std::min_element(data.begin(), data.end(), p_y_cmp);
	
	double norm_x = data[data.size() - 1].x - data[0].x, norm_y = 0.202392 - (*min_y).y;
	int inner_w = width - padding * 2, inner_h = height - padding * 2;
	for (unsigned int i = 1; i < data.size(); ++i){
		cv::line(plot_show,
			cv::Point(padding + round((data[i - 1].x - data[0].x) / norm_x * inner_w), padding + inner_h - round((data[i - 1].y - (*min_y).y) / norm_y * inner_h)),
			cv::Point(padding + round((data[i].x - data[0].x) / norm_x * inner_w), padding + inner_h - round((data[i].y - (*min_y).y) / norm_y * inner_h)),
			cv::Scalar(200, 200, 0),
			2);
	}
	printf("max_y: %lf, min_y: %lf\n", (*max_y).y, (*min_y).y);
	cv::imshow("plot", plot_show);
	cv::imwrite("results/plot.png", plot_show);
}