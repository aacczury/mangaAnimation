#ifndef _COLOR_CHIPS_H_
#define _COLOR_CHIPS_H_

#include <opencv2/core/core.hpp>
#include <vector>

class normal_color{
public:
	cv::Scalar white = cv::Scalar(255, 255, 255);
	cv::Scalar gray = cv::Scalar(128, 128, 128);
	cv::Scalar black = cv::Scalar(0, 0, 0);

	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar yellow = cv::Scalar(0, 255, 255);
	cv::Scalar lime = cv::Scalar(0, 255, 0);
	cv::Scalar cyan = cv::Scalar(255, 255, 0);
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar magenta = cv::Scalar(255, 0, 255);

	cv::Scalar maroon = cv::Scalar(0, 0, 128);
	cv::Scalar olive = cv::Scalar(0, 128, 128);
	cv::Scalar green = cv::Scalar(0, 128, 0);
	cv::Scalar teal = cv::Scalar(128, 128, 0);
	cv::Scalar navy = cv::Scalar(128, 0, 0);
	cv::Scalar purple = cv::Scalar(128, 0, 128);

	cv::Scalar brown = cv::Scalar(42, 42, 165);
	cv::Scalar orange = cv::Scalar(0, 165, 255);
	cv::Scalar turquoise = cv::Scalar(200, 213, 48);
	cv::Scalar azure = cv::Scalar(255, 127, 0);
	cv::Scalar mauve = cv::Scalar(255, 64, 102);
	cv::Scalar violet = cv::Scalar(255, 0, 139);

	std::vector<cv::Scalar> color_chips = std::vector<cv::Scalar>{
		red, yellow, lime, cyan, blue, magenta,
		maroon, olive, green, teal, navy, purple,
		brown, orange, turquoise, azure, mauve, violet};
};

class material_color{
public:
	cv::Scalar white = cv::Scalar(255, 255, 255);
	cv::Scalar gray = cv::Scalar(158, 158, 158);
	cv::Scalar black = cv::Scalar(0, 0, 0);

	cv::Scalar red = cv::Scalar(54, 67, 244);
	cv::Scalar yellow = cv::Scalar(59, 235, 255);
	cv::Scalar lime = cv::Scalar(57, 220, 205);
	cv::Scalar cyan = cv::Scalar(212, 188, 0);
	cv::Scalar blue = cv::Scalar(243, 150, 33);
	cv::Scalar purple = cv::Scalar(176, 39, 156);

	cv::Scalar pink = cv::Scalar(99, 30, 233);
	cv::Scalar amber = cv::Scalar(7, 193, 255);
	cv::Scalar green = cv::Scalar(80, 175, 76);
	cv::Scalar teal = cv::Scalar(136, 150, 0);
	cv::Scalar indigo = cv::Scalar(181, 81, 63);
	cv::Scalar deep_purple = cv::Scalar(183, 58, 103);

	cv::Scalar brown = cv::Scalar(72, 85, 121);
	cv::Scalar orange = cv::Scalar(0, 152, 255);
	cv::Scalar light_green = cv::Scalar(74, 195, 139);
	cv::Scalar light_blue = cv::Scalar(244, 169, 3);
	cv::Scalar blue_gray = cv::Scalar(139, 125, 96);
	cv::Scalar deep_orange = cv::Scalar(34, 87, 255);

	std::vector<cv::Scalar> color_chips = std::vector<cv::Scalar>{
		red, yellow, lime, cyan, blue, purple,
		pink, amber, green, teal, indigo, deep_purple,
		brown, orange, light_green, light_blue, blue_gray, deep_orange};
};

class deep_color{
public:
	cv::Scalar white = cv::Scalar(255, 255, 255);
	cv::Scalar gray = cv::Scalar(100, 100, 100);
	cv::Scalar black = cv::Scalar(0, 0, 0);

	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar yellow = cv::Scalar(0, 150, 150);
	cv::Scalar lime = cv::Scalar(0, 255, 0);
	cv::Scalar cyan = cv::Scalar(150, 150, 0);
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar magenta = cv::Scalar(150, 0, 150);

	cv::Scalar maroon = cv::Scalar(0, 0, 100);
	cv::Scalar olive = cv::Scalar(0, 75, 75);
	cv::Scalar green = cv::Scalar(0, 100, 0);
	cv::Scalar teal = cv::Scalar(75, 75, 0);
	cv::Scalar navy = cv::Scalar(100, 0, 0);
	cv::Scalar purple = cv::Scalar(75, 0, 75);

	cv::Scalar brown = cv::Scalar(42, 42, 165);
	cv::Scalar orange = cv::Scalar(0, 165, 255);
	cv::Scalar turquoise = cv::Scalar(200, 213, 48);
	cv::Scalar azure = cv::Scalar(255, 127, 0);
	cv::Scalar mauve = cv::Scalar(255, 64, 102);
	cv::Scalar violet = cv::Scalar(255, 0, 139);

	std::vector<cv::Scalar> color_chips = std::vector<cv::Scalar>{
		red, yellow, lime, cyan, blue, magenta,
			maroon, olive, green, teal, navy, purple,
			brown, orange, turquoise, azure, mauve, violet};
};

#endif