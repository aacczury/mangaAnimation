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

void mangaShow::compute_curvature(std::vector<cv::Point> curve){
	std::vector<double> curvature;
	for (unsigned int i = 1; i < curve.size() - 2; ++i){
		double dx1 = curve[i].x - curve[i - 1].x;
		double dx2 = curve[i + 1].x - curve[i].x;
		double dy1 = curve[i].y - curve[i - 1].y;
		double dy2 = curve[i + 1].y - curve[i].y;
		double ddx = dx2 - dx1;
		double ddy = dy2 - dy1;

		if (pow(dx1 * dx1 + dy1 * dy1, 1.5) == 0) curvature.push_back(0);
		else curvature.push_back((dx1 * ddy - dy1 * ddx) / pow(dx1 * dx1 + dy1 * dy1, 1.5));
	}
}