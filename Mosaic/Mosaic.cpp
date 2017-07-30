#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

void mosaicSimple(const cv::Mat& src, cv::Size size, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_8UC3);
    CV_Assert(size.width <= src.cols && size.height <= src.rows);

    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_8UC3);

    int numHori = (cols + size.width - 1) / size.width;
    int numVert = (rows + size.height - 1) / size.height;

    for (int i = 0; i < numVert; i++)
    {
        int yBeg = i * size.height, yEnd = yBeg + size.height;
        yEnd = std::min(rows, yEnd);

        for (int j = 0; j < numHori; j++)
        {
            int xBeg = j * size.width, xEnd = xBeg + size.width;
            xEnd = std::min(cols, xEnd);

            // Call OpenCV interfaces to realize the above two for loops
            cv::Rect roi(cv::Point(xBeg, yBeg), cv::Point(xEnd, yEnd));
            cv::Mat srcPart = src(roi);
            cv::Scalar mu = cv::mean(srcPart);
            cv::Mat dstPart = dst(roi);
            dstPart.setTo(mu);
        }
    }
}

void mosaic(const cv::Mat& src, cv::Size size, cv::Mat& dst)
{
    CV_Assert(src.data && src.type() == CV_8UC3);
    CV_Assert(size.width <= src.cols && size.height <= src.rows);

    int rows = src.rows, cols = src.cols;
    dst.create(rows, cols, CV_8UC3);

    int numHori = (cols + size.width - 1) / size.width;
    int numVert = (rows + size.height - 1) / size.height;

    for (int i = 0; i < numVert; i++)
    {
        int yBeg = i * size.height, yEnd = yBeg + size.height;
        yEnd = std::min(rows, yEnd);

        for (int j = 0; j < numHori; j++)
        {
            int xBeg = j * size.width, xEnd = xBeg + size.width;
            xEnd = std::min(cols, xEnd);

            int accumB = 0, accumG = 0, accumR = 0;
            for (int u = yBeg; u < yEnd; u++)
            {
                const unsigned char* ptr = src.ptr<unsigned char>(u) + xBeg * 3;
                for (int v = xBeg; v < xEnd; v++)
                {
                    accumB += *(ptr++);
                    accumG += *(ptr++);
                    accumR += *(ptr++);
                }
            }
            
            double invScale = 1.0 / ((yEnd - yBeg) * (xEnd - xBeg));
            int b = accumB * invScale, g = accumG * invScale, r = accumR * invScale;
            for (int u = yBeg; u < yEnd; u++)
            {
                unsigned char* ptr = dst.ptr<unsigned char>(u) + xBeg * 3;
                for (int v = xBeg; v < xEnd; v++)
                {
                    *(ptr++) = b;
                    *(ptr++) = g;
                    *(ptr++) = r;
                }
            }
        }
    }
}

int main1()
{
    cv::Mat src = cv::imread("default3.jpg");
    cv::Mat dst(src.size(), CV_8UC3);
    cv::Size s(8, 8);

    long long int beg, end;
    double freq = cv::getTickFrequency();

    beg = cv::getTickCount();
    for (int i = 0; i < 100; i++)
        mosaicSimple(src, s, dst);
    end = cv::getTickCount();
    printf("time = %f\n", (end - beg) / freq);

    beg = cv::getTickCount();
    for (int i = 0; i < 100; i++)
        mosaic(src, s, dst);
    end = cv::getTickCount();
    printf("time = %f\n", (end - beg) / freq);

    return 0;
}

cv::Mat orig;
cv::Mat mask0, mask1, mask2;
cv::Mat proc;
cv::Mat show;
cv::Mat kern;
int cellSize = 8;
int radius = 16;
int lastx, lasty;

void changeCellSize(int val, void* data)
{
    cellSize = val;
    if (cellSize <= 0) cellSize = 1;
    mosaic(orig, cv::Size(cellSize, cellSize), proc);
    proc.copyTo(show, mask0);
    cv::imshow("show", show);
}

void changeRadius(int val, void* data)
{
    radius = val;
    if (radius <= 0) radius = 1;
    kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
}

void mouse(int event, int x, int y, int flags, void *data)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        lastx = x;
        lasty = y;
        cv::circle(mask0, cv::Point(x, y), radius, 255, -1);
        proc.copyTo(show, mask0);
        cv::imshow("show", show);
    }
    if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON))
    {
        cv::Rect r(cv::Point(lastx, lasty), cv::Point(x, y));
        r.x -= radius + 1;
        r.y -= radius + 1;
        r.width += radius * 2 + 2;
        r.height += radius * 2 + 2;
        cv::Rect base(0, 0, orig.cols, orig.rows);
        r &= base;

        cv::Mat mask0P = mask0(r);
        cv::Mat mask1P = mask1(r);
        cv::Mat mask2P = mask2(r);
        cv::Mat procP = proc(r);
        cv::Mat showP = show(r);

        cv::line(mask1, cv::Point(lastx, lasty), cv::Point(x, y), 255);
        cv::dilate(mask1P, mask2P, kern);
        mask0P |= mask2P;
        procP.copyTo(showP, mask0P);

        mask1P.setTo(0);
        mask2P.setTo(0);
        
        lastx = x;
        lasty = y;

        cv::imshow("show", show);
    }
}

int main()
{
    orig = cv::imread("default3.jpg", cv::IMREAD_COLOR);
    if (!orig.data)
        return 1;

    mosaic(orig, cv::Size(cellSize, cellSize), proc);
    kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));

    orig.copyTo(show);
    mask0 = cv::Mat::zeros(orig.size(), CV_8UC1);
    mask1 = cv::Mat::zeros(orig.size(), CV_8UC1);
    mask2 = cv::Mat::zeros(orig.size(), CV_8UC1);

    cv::namedWindow("show");
    cv::createTrackbar("cell size", "show", &cellSize, 32, changeCellSize);
    cv::createTrackbar("radius", "show", &radius, 128, changeRadius);
    cv::setMouseCallback("show", mouse);
    cv::imshow("show", show);
    int count = 0;
    while (true)
    {
        int k = cv::waitKey(0);
        if (k == 'q')
            break;
        if (k == 's')
        {
            char buf[256];
            sprintf(buf, "save%d.jpg", count++);
            cv::imwrite(buf, show);
        }
        if (k == 'e')
        {
            mask0.setTo(0);
            orig.copyTo(show);
            cv::imshow("show", show);
        }
    }

    return 0;
}