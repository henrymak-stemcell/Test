#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>

using namespace cv;

int main(void)
{
	Mat im = imread("b1.png", CV_LOAD_IMAGE_GRAYSCALE);

	SimpleBlobDetector::Params params;

	params.filterByColor = true;
	params.blobColor = 0;

	params.minThreshold = 95;
	params.maxThreshold = 255;

	params.filterByArea = true;
	params.minArea = 5;

	params.filterByCircularity = true;
	params.minCircularity = 0.5;

	params.filterByConvexity = true;
	params.minConvexity = 0.5;

	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	std::vector<KeyPoint> keypoints;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	detector->detect(im, keypoints);

	std::ofstream coordinateFile("coordinateFile.txt");
	
	for (int i = 0; i < keypoints.size(); i++)
	{
		coordinateFile << keypoints[i].pt.x << "," << keypoints[i].pt.y << std::endl;
	}

	Mat im_with_keypoints;
	drawKeypoints(im, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	std::cout << keypoints.size() << std::endl;

	imshow("keypoints", im_with_keypoints);
	waitKey(0);

	return 0;
}