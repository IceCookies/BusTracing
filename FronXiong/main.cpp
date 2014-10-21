#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

Mat image;

bool backprojMode = false;
bool selectObject = false;
char fileName[100];
char fileName1[100];

vector<Point> points;
vector<Point> points1;



int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection,selection1;
vector<Rect> selections;
int vmin = 10, vmax = 256, smin = 30;

static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, image.cols, image.rows);

    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;

    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject -=1;
        selections.push_back(selection);
        cout<<"Track Window "<<selections.size()<<endl;
        break;
    }
}


int main()
{

    CvCapture *cap = cvCreateFileCapture("test1.mp4");
    Rect trackWindow,trackWindow1;
    int hsize = 16;

    float hranges[] = {0,180};
    const float* phranges = hranges;
    namedWindow( "Histogram", 1 );
    namedWindow( "CamShift Demo", 1 );
    namedWindow( "show", 1 );
    sprintf(fileName, "1");
    sprintf(fileName1, "2");

    setMouseCallback( "CamShift Demo", onMouse, 0 );


    Mat frame,line, hsv, hue,hue1, mask, hist, hist1,histimg = Mat::zeros(600, 550, CV_8UC3), backproj,backproj1;

    for(;;)
    {
            frame = cvQueryFrame(cap);
            if( frame.empty() )
                break;
        frame.copyTo(image);
        frame.copyTo(line);
        line.setTo(0);
            cvtColor(image, hsv, CV_BGR2HSV);

            if( trackObject == -2 )
            {
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                hue1.create(hsv.size(), hsv.depth());

                mixChannels(&hsv, 1, &hue, 1, ch, 1);      //分离出hue分量

                mixChannels(&hsv, 1, &hue1, 1, ch, 1);      //分离出hue分量


                if( trackObject < 0 )
                {
                    Mat roi1(hue1, selections[1]), maskroi1(mask, selections[1]);
                    Mat roi(hue, selections[0]), maskroi(mask, selections[0]);
                    calcHist(&roi1, 1, 0, maskroi1, hist1, 1, &hsize, &phranges);            //计算selection内的hue分量的hist

                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);            //计算selection内的hue分量的hist
                    normalize(hist, hist, 0, 255, CV_MINMAX);                             // 归一化

                    normalize(hist1, hist1, 0, 255, CV_MINMAX);                             // 归一化


                    trackWindow = selections[0];
                    trackWindow1 = selections[1];

                    trackObject = -2;
                }

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);                  //反向投影
                calcBackProject(&hue1, 1, 0, hist1, backproj1, &phranges);                  //反向投影

                backproj &= mask;

                backproj1 &= mask;

                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                    TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                RotatedRect trackBox1 = CamShift(backproj1, trackWindow1,
                                    TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));


                if( trackWindow.area() <= 1 )
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                if( trackWindow1.area() <= 1 )
                {
                    int cols = backproj1.cols, rows = backproj1.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow1 = Rect(trackWindow1.x - r, trackWindow1.y - r,
                                       trackWindow1.x + r, trackWindow1.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                if( backprojMode ){
                    cvtColor( backproj, image, CV_GRAY2BGR );
                 cvtColor( backproj1, image, CV_GRAY2BGR );
                }
                ellipse( image, trackBox, Scalar(0,0,0), 3, CV_AA );

                ellipse( image, trackBox1, Scalar(0,0,255), 3, CV_AA );


                cv::putText(image,fileName,trackBox.center,1,3,Scalar(0,0,255),3,8,false);
                cv::putText(image,fileName1,trackBox1.center,1,3,Scalar(0,0,255),3,8,false);

                circle(image, trackBox.center,5,Scalar(255,0,0),1,8,0);
                circle(image, trackBox1.center,5,Scalar(255,0,0),1,8,0);


                points.push_back(trackBox.center);
                points1.push_back(trackBox1.center);


                for(int i=0;i<points.size();i++)
                {
                    circle(image, points[i],4,Scalar(0,255,0),1,8,0);
                    circle(line, points[i],4,Scalar(0,255,0),1,8,0);
                    cv::putText(line,fileName,trackBox.center,1,3,Scalar(0,0,255),3,8,false);

                }


                for(int i=0;i<points1.size();i++)
                {
                    circle(image, points1[i],4,Scalar(255,0,0),1,8,0);
                    circle(line, points1[i],4,Scalar(255,0,0),1,8,0);
                    cv::putText(line,fileName1,trackBox1.center,1,3,Scalar(0,0,255),3,8,false);
                }

                ofstream fout("track.txt",ios::app);
                   if(!fout)
                   {
                       cout<<"!";
                       exit(1);
                   }
                   fout<<(int)(trackBox.center.x+0.5)<<" "<<(int)(trackBox.center.y+0.5)<<" ";
            }

        if( selectObject && selection.width > 0 && selection.height > 0 )
        {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }


        imshow( "CamShift Demo", image );
        imshow( "Histogram", histimg );
        imshow("show",line);
        char c = (char)waitKey(2);
        if( c == 27 )
            break;
    }

    return 0;
}
