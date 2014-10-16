// Tran.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <stdlib.h>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <boost/format.hpp>
#include <boost/timer.hpp>
#include <boost/thread.hpp>
#include <boost/program_options.hpp>

#include <libzplay.h>

#include "Geology.hpp"

#define DEFINED_ALLREADY

using namespace cv;
using namespace libZPlay;
using namespace boost::program_options;

bool _bug_flag = false;
	

Mat getLUT()
{
	Mat lut = Mat::zeros(1,256,CV_8UC1);
	for(long i=0;i<=255;i++)
	{
		uchar newValue = 0;
		if(i<=80)
		{
			newValue = (uchar)((double)i * 0.625);
		}
		else if(i<=160)
		{
			newValue = (uchar)((double)i * 1.875 - 100);
		}
		else
		{
			newValue = (uchar)((double)i * 55.0 / 95.0 + 200 - 160*55.0 / 95.0);
		}
		lut.at<uchar>(i) = newValue;

	}
	return lut;
}

void OutPutKeyPoint(std::vector<KeyPoint> keypoints_object,const string fileName)
{
	ofstream of(fileName);
	of << "Point=" << keypoints_object.size() << endl;
	
	for(int i=0;i<keypoints_object.size();i++)
	{
		of << i << "\t" << keypoints_object[i].pt.x << "\t" << keypoints_object[i].pt.y << endl;
	}

	of.close();
}

void OutPutKeyMat(Mat img,const string fileName)
{
	
	ofstream of(fileName);
	int count =0;
	for(auto it = img.begin<uchar>();it!=img.end<uchar>();it++)
	{
		if((*it)>100)
		{
			count ++;
		}
	}
	of << "Point=" << count << endl;
	
	count = 0;
	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			uchar it = img.at<uchar>(i,j);
			if(it>100)
			{
				of << count << "\t" << i << "\t" << j << endl;
				count ++;
			}
		}
	}

	of.close();
}

int OtsuValue(const Mat& img)
{    
	int rows=img.rows;    
    int cols=img.cols;        
    long size = rows * cols;   
  
    //histogram    
    float histogram[256] = {0};    
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			const uchar& p = img.at<uchar>(i,j);
			histogram[int(p)]++;
		}
	}  
  
    int threshold;      
    long sum0 = 0, sum1 = 0; //存储前景的灰度总和和背景灰度总和  
    long cnt0 = 0, cnt1 = 0; //前景的总个数和背景的总个数  
    double w0 = 0, w1 = 0; //前景和背景所占整幅图像的比例  
    double u0 = 0, u1 = 0;  //前景和背景的平均灰度  
    double variance = 0; //最大类间方差  
    int i, j;  
    double u = 0;  
    double maxVariance = 0;  
    for(i = 1; i < 256; i++) //一次遍历每个像素  
    {    
        sum0 = 0;  
        sum1 = 0;   
        cnt0 = 0;  
        cnt1 = 0;  
        w0 = 0;  
        w1 = 0;  
        for(j = 0; j < i; j++)  
        {  
            cnt0 += histogram[j];  
            sum0 += j * histogram[j];  
        }  
  
        u0 = (double)sum0 /  cnt0;   
        w0 = (double)cnt0 / size;  
  
        for(j = i ; j <= 255; j++)  
        {  
            cnt1 += histogram[j];  
            sum1 += j * histogram[j];  
        }  
  
        u1 = (double)sum1 / cnt1;  
        w1 = 1 - w0; // (double)cnt1 / size;  
  
        u = u0 * w0 + u1 * w1; //图像的平均灰度  
        //variance =  w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);  
        variance =  w0 * w1 *  (u0 - u1) * (u0 - u1);  
        if(variance > maxVariance)   
        {    
            maxVariance = variance;    
            threshold = i;    
        }   
    }    
  
    //printf("threshold = %d\n", threshold);  
    return threshold;    
}    

double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
 double dx1 = pt1.x - pt0.x;
 double dy1 = pt1.y - pt0.y;
 double dx2 = pt2.x - pt0.x;
 double dy2 = pt2.y - pt0.y;
 return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

Mat element(int size)
{
	return getStructuringElement(MORPH_RECT,Size(size,size));
}

class ClassiferHelper
{
public:
	ClassiferHelper(int sampleCount);
	inline int getSampleCount(){return this->sampleCount;};
	Mat getCharacterImage(Mat img_scene);
	void calcPixelSum();
	pair<int,double> Score(Mat mask);
	Mat AffineImage(Mat img_scene , Mat clip);

	std::vector<Mat> img_objects; 
	std::vector<Mat> img_map_objects; 
	std::vector<double> pixSum;

private:
	const int TemplateSize;
	int sampleCount;

	Point2f srcTri[3];
	Mat mask;
};
ClassiferHelper::ClassiferHelper(int sampleCount):TemplateSize(100)
{
	this->sampleCount = sampleCount;
}
Mat ClassiferHelper::getCharacterImage(Mat img_scene)
{
	//Mat sc1,sc2;
	Mat HSV,H,S,V;
	std::vector<Mat> hsv_v;
	hsv_v.push_back(H);
	hsv_v.push_back(S);
	hsv_v.push_back(V);
	////2-预处理图像////
	//调整大小
	//double s = 400.0 / img_scene.rows;
	//resize(img_scene,img_scene,Size(),s,s);

	//颜色空间转换
	//cvtColor(img_scene_1,img_scene_2,CV_RGB2GRAY);
	cvtColor(img_scene,HSV,CV_RGB2HSV);
	split(HSV,hsv_v);
	blur(hsv_v[1],S,Size(7,7));
	Mat V_S = hsv_v[2]-hsv_v[1];
	//Mat V_S = hsv_v[2]-S;
	//Mat V_S = hsv_v[2];
	//threshold(hsv_v[0],hsv_v[0],85,255,THRESH_TOZERO);
	//threshold(hsv_v[0],hsv_v[0],120,255,THRESH_TOZERO_INV);
	//threshold(hsv_v[0],hsv_v[0],50,255,THRESH_BINARY);
	//imshow("0",hsv_v[0]);
	//imshow("1",hsv_v[1]);
	//imshow("2",hsv_v[2]);
	//imshow("1",S);
	//imshow("V-S",V_S);
	//特征增强
	Mat obj_img;
	V_S.copyTo(obj_img);
	//Laplacian(V_S,obj_img,-1,3);
	Canny(V_S,obj_img,150,250);
	if(_bug_flag)
	imshow("obj_img",obj_img);
	
	//int key = waitKey(100);
	//if(key!=-1)
	//{
	//	imwrite("H.bmp",hsv_v[0]);
	//	imwrite("S.bmp",hsv_v[1]);
	//	imwrite("V.bmp",hsv_v[2]); 
	//}

	dilate(obj_img,obj_img,element(7));
	vector<vector<Point> > contours0;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	findContours(obj_img,contours0,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
	mask = Mat::zeros(obj_img.rows,obj_img.cols,CV_8UC1);
	contours.resize(contours0.size());
	size_t index = -1;
	double minS = 0;
	for( size_t k = 0; k < contours0.size(); k++ )
	{
		approxPolyDP(Mat(contours0[k]), contours[k], arcLength(contours0[k],true)*0.02, true);
	}
	for( size_t k = 0; k < contours0.size(); k++ )
	{
		double s = 0;
		double L = arcLength(contours[k],true);
		double A = contourArea(contours[k]);
		if(contours[k].size()==4 && isContourConvex(contours[k]) && A>3600)
		{
			for(int i=0;i<4;i++)
			{
				double t = angle(contours[k][i%4],contours[k][(i+1)%4],contours[k][(i+1)%4]);
				s = (t > s)? t : s;
			}
			//cout << L << "^2/" << A << "=" << L*L/A << endl;
			double sq = L*L/A;
			if(s<0.1 && sq >15 && sq < 17)
			{
				if(index==-1 || minS > A)
				{
					index = k;
					minS = A;
				}
			}
		}
	}
	if(index==-1)
	{
		return Mat();
	}

	Point pts[1][4] ;
	pts[0][0]= contours[index][0];
	pts[0][1]= contours[index][1];
	pts[0][2]= contours[index][2];
	pts[0][3]= contours[index][3];
	const Point* ppt[1] = {pts[0]};
	int npts = 4;
	//drawContours(obj_img,contours,index,Scalar::all(125));
	fillPoly(mask,ppt,&npts,1,Scalar::all(255));
	if(_bug_flag)
	imshow("mask",mask);
	//图像加掩板
	Mat img_scene_mask = Mat::zeros(img_scene.rows,img_scene.cols,CV_8UC1);
	V_S.copyTo(img_scene_mask,mask);
	if(_bug_flag)
	imshow("img_scene_mask",img_scene_mask);
	
	Point2f dstTri[3];

	srcTri[0] = Point2f( contours[index][0].x,contours[index][0].y );
	srcTri[1] = Point2f( contours[index][1].x,contours[index][1].y );
	srcTri[2] = Point2f( contours[index][3].x,contours[index][3].y );

	dstTri[0] = Point2f( 0,0 );
	dstTri[1] = Point2f( TemplateSize - 1, 0 );
	dstTri[2] = Point2f( 0, TemplateSize - 1 );

	Mat warp_mat( 2, 3, CV_32FC1 );
	warp_mat = getAffineTransform( srcTri, dstTri );
	
	Mat warp_dst;
	warp_dst = Mat::zeros( TemplateSize, TemplateSize, img_scene_mask.type() );
	warpAffine( img_scene_mask, warp_dst, warp_mat, warp_dst.size() );
	int otsuv = OtsuValue(warp_dst);
	threshold(warp_dst,warp_dst,otsuv,255,THRESH_BINARY_INV);
	//imshow("img_scene_mask",warp_dst);
	return warp_dst;
}
void ClassiferHelper::calcPixelSum()
{
	for(int i=0;i<sampleCount;i++)
	{
		pixSum.push_back(cv::sum(img_objects[i])[0]);
	}
}
pair<int,double> ClassiferHelper::Score(Mat mask)
{
	
	Mat pixel;
	
	double maxScore = 0;
	int ScoreIndex = 0;

	//system("cls");
	for(int i=0;i<getSampleCount();i++)
	{
		//cout << i << ":";
		pixel = Mat::zeros(img_objects[i].rows,img_objects[i].cols,CV_8UC1);
		img_objects[i].copyTo(pixel,mask);
		int PiexlCount2 = cv::sum(pixel)[0];
		//if(_bug_flag)
		//cout << PiexlCount2 << "," << pixSum[i] << "," << PiexlCount2/pixSum[i] << endl;
		double score = PiexlCount2/(pixSum[i]+cv::sum(mask)[0]-PiexlCount2);
		if(score>maxScore)
		{
			maxScore = score;
			ScoreIndex = i;
		}
	}
	//waitKey();
	return pair<int,double>(ScoreIndex,maxScore);
}
Mat ClassiferHelper::AffineImage(Mat img_scene , Mat clip)
{
	Point2f srcTri2[3];

	srcTri2[0] = Point2f( 0,0 );
	srcTri2[1] = Point2f( clip.cols - 1, 0 );
	srcTri2[2] = Point2f( 0, clip.rows - 1 );

	Mat img_scene_tmp = Mat::zeros(img_scene.rows,img_scene.cols,img_scene.type());
	Mat warp_mat( 2, 3, CV_32FC1 );
	warp_mat = getAffineTransform( srcTri2 , srcTri );
	warpAffine( clip, img_scene_tmp, warp_mat, img_scene_tmp.size() );
	Mat img_scene_tmp2;
	img_scene.copyTo(img_scene_tmp2,255-mask);
	return img_scene_tmp+img_scene_tmp2;
}

typedef struct _VideoCaptureStruct
{
	VideoCapture* cap;
	int index;
	bool flag;
	//string indexstr;
}VideoCaptureStruct;

VideoCaptureStruct* now_play = NULL;
ZPlay * now_play_audio = NULL;
ZPlay * last_play_audio = NULL;
//VideoCaptureStruct* last_play = NULL;

bool isEndOfApp = false;
boost::mutex mutex_of_isEndOfApp;
boost::mutex mutex_of_nowPlay;
boost::mutex mutex_of_nowPlayAudio;
boost::mutex mutex_of_lastPlayAudio;
vector<VideoCaptureStruct*> videoRecopy;
vector<VideoCaptureStruct*> videos;
vector<ZPlay *> audios;

int VideoDelay = 25;


void reopen(VideoCaptureStruct* video,VideoCaptureStruct* changeto=NULL)
{
	video->cap->set(CV_CAP_PROP_POS_MSEC,5);
	if(changeto!=NULL)
	{
		video = changeto;
	}
}

void play()
{
	boost::timer timer;
	Mat frame_play;
	bool isEndOfAppCopy = false;
	VideoCaptureStruct* now_play_Copy = NULL;
	ZPlay* now_play_audio_Copy = NULL;
	ZPlay* last_play_audio_Copy = NULL;
	int key = -1;
	do{
		try{
			//mutex_of_isEndOfApp.lock();
			//mutex_of_nowPlay.lock();
			isEndOfAppCopy = isEndOfApp;
			now_play_Copy = now_play;
			now_play_audio_Copy = now_play_audio;
			last_play_audio_Copy = last_play_audio;
			//mutex_of_isEndOfApp.unlock();
			//mutex_of_nowPlay.unlock();
		}
		catch(...)
		{
			//mutex_of_isEndOfApp.unlock();
			//mutex_of_nowPlay.unlock();
		}
		
		if(now_play_audio_Copy != last_play_audio_Copy)
		{
			if(last_play_audio_Copy!=NULL)
			{
				last_play_audio_Copy->Pause();
			}
			if(now_play_audio_Copy!=NULL)
			{
				TStreamTime time;
				time.samples = 0;
				TStreamInfo info;
				now_play_audio_Copy->GetStreamInfo(&info);
				now_play_audio_Copy->Play();
				//now_play_audio_Copy->PlayLoop(libZPlay::tfSamples,&time,libZPlay::tfHMS,&info.Length,1,1);
			}
		}
		double time1 = timer.elapsed();
		if(now_play_Copy!=NULL)
		{
			(*now_play_Copy->cap) >> frame_play;
			if(frame_play.empty())
			{
				VideoCaptureStruct* tmp = now_play_Copy;
				now_play_Copy = videoRecopy[tmp->index-1];
				try{
					//mutex_of_nowPlay.lock();
					isEndOfAppCopy = isEndOfApp;
					now_play = now_play_Copy;
					//mutex_of_nowPlay.unlock();
				}
				catch(...)
				{
					//mutex_of_nowPlay.unlock();
				}
				videos[tmp->index-1] = now_play_Copy;
				videoRecopy[tmp->index-1] = tmp;
				//boost::format f("./video/%d.mp4");
				//f % (now_play_Copy->index);
				(*now_play_Copy->cap) >> frame_play;
				VideoCaptureStruct* nullPtr = NULL;
				boost::thread(reopen,tmp,nullPtr);
			}
		}
		if(!frame_play.empty())
		{
			imshow("video",frame_play);
			TStreamTime time;
			now_play_audio_Copy->GetPosition(&time);
			int VTime = (int)(now_play_Copy->cap->get(CV_CAP_PROP_POS_MSEC)) ;
			int AVdelay = VTime - time.ms;
			if(AVdelay > 1000 || AVdelay <-1000)
			{
				if(_bug_flag)
				{
					cout << "A:" << time.ms << " | " << "V:" << VTime << endl;
				}
				TStreamTime new_time;
				new_time.ms = VTime;
				now_play_audio_Copy->Seek(TTimeFormat::tfMillisecond,&new_time,AVdelay>0?TSeekMethod::smFromCurrentForward:TSeekMethod::smFromCurrentBackward);
			}
			//cout << (int)(now_play_Copy->cap->get(CV_CAP_PROP_POS_MSEC)) << "|" << time.ms << endl;
			//cout << now_play_Copy->flag <<endl;
		}
		double time2 = timer.elapsed();
		/*if(now_play_Copy!=last_play)
		{
			boost::thread(reopen,last_play,now_play_Copy);
		}*/
		int delay = VideoDelay +(int)(time1 - time2);
		key = waitKey(delay>10?delay:10);
		timer.restart();
		time1 = timer.elapsed();
	}
	while(!isEndOfAppCopy);
}

void PlayBackMusic(ZPlay* player)
{
	player->Play();
	while(!isEndOfApp)
    {
		TStreamStatus status;
		player->GetStatus(&status); 
		if(status.fPlay == 0)
		{
			player->Resume();
			player->Play();
		}
		Sleep(300);

	}

}

int main( int argc, char** argv )
{
	string videoSuf;
	string audioSuf;
	string videoPath;
	string audioPath;
	string backgroundMusic;
	
	 //一共有6*8图像
	int num_of_char_type = 6;
	int num_of_char_pose = 8;
	int backgroungMusicVol = 50;
	int audioVol = 100;

	options_description opts("参数");

	opts.add_options()
		("help",																"帮助")
		("debug",																"调试模式")
		("bgMusic",		value<string>(&backgroundMusic),						"背景音乐文件")
		("bgMusicVol",	value<int>(&backgroungMusicVol)->default_value(50),		"背景音乐音量 0-100")
		("audioVol",	value<int>(&audioVol)->default_value(100),				"前景音乐音量 0-100")
		("videoDelay",	value<int>(&VideoDelay)->default_value(25),				"视频延迟时间")
		("videoSuf",	value<string>(&videoSuf)->default_value("mp4"),			"视频文件后缀名")
		("audioSuf",	value<string>(&audioSuf)->default_value("mp3"),			"音频文件后缀名")
		("videoPath",	value<string>(&videoPath)->default_value("./video/"),	"视频文件路径")
		("audioPath",	value<string>(&audioPath)->default_value("./audio/"),	"音频文件路径")
		("charType",	value<int>(&num_of_char_type)->default_value(6),		"字符种类数量")
		("charPose",	value<int>(&num_of_char_pose)->default_value(8),		"每个字符姿态数量")
		
		;
	variables_map vm;
	store(parse_command_line(argc,argv,opts),vm);
	notify(vm);
	if(vm.count("help"))
	{
		cout << opts << endl;
		return 1;
	}
	//cout << "Arc" << argc << endl;
	if(vm.count("debug"))
	{
		_bug_flag = true;
	}

	if(vm.count("bgMusic"))
	{
		if(_bug_flag)
			cout << "打开背景音乐文件" << backgroundMusic << endl;
		ZPlay* player = CreateZPlay();
		int openResult = player->OpenFile(backgroundMusic.c_str(),sfAutodetect);
		if(_bug_flag)
			cout << "背景音乐打开" << ((openResult!=0)?"成功":"失败") << endl;
		player->SetPlayerVolume(backgroungMusicVol,backgroungMusicVol);
		//player->Play();
		boost::thread(PlayBackMusic,player);
	}
	int index_of_remark = 0;
	ClassiferHelper classifer(num_of_char_type*num_of_char_pose);

	for(int i=1;i<=classifer.getSampleCount()/num_of_char_pose;i++)
	{
		boost::format f("./sample/%d.bmp");
		f % i;
		Mat obj_img_map_in = imread( f.str()  );
		classifer.img_map_objects.push_back(obj_img_map_in);
	}
	for(int i=1;i<=classifer.getSampleCount();i++)
	{
		//读取图像
		boost::format f("./sample/%d-%d.bmp");
		f %((i-1)/num_of_char_pose + 1) % ((i)%num_of_char_pose);
		Mat obj_img_in = imread( f.str() , CV_LOAD_IMAGE_GRAYSCALE );
		threshold(obj_img_in,obj_img_in,200,255,THRESH_BINARY);
		classifer.img_objects.push_back(obj_img_in);
	}
	classifer.calcPixelSum();
	
    int ids[] = {-1,-1,-1};
    int index = 0;
    int nowindex = -1;

	for(int i=0;i<num_of_char_type;i++)
	{
		boost::format f(videoPath+"/%d."+videoSuf);
		f % (i+1);
		if(_bug_flag)
			cout << "打开视频" << f.str() << endl;
		VideoCaptureStruct* capstruct = (VideoCaptureStruct*)malloc(sizeof(VideoCaptureStruct));
		capstruct->cap = new VideoCapture(f.str());
		capstruct->index = (i+1);
		capstruct->flag = true;
		//capstruct->indexstr = "./video/%d.mp4";
		videos.push_back(capstruct);

		boost::format f2(videoPath+"%d-1."+videoSuf);
		f2 % (i+1);
		VideoCaptureStruct* capstruct2 = (VideoCaptureStruct*)malloc(sizeof(VideoCaptureStruct));
		capstruct2->cap = new VideoCapture(f2.str());
		capstruct2->index = (i+1);
		capstruct2->flag = false;
		//capstruct2->indexstr = "./video/%d-1.mp4";
		videoRecopy.push_back(capstruct2);
	}
	
	for(int i=0;i<num_of_char_type;i++)
	{
		boost::format f(audioPath+"/%d."+audioSuf);
		f % (i+1);
		if(_bug_flag)
			cout << "打开音频" << f.str() << endl;
		ZPlay* player = CreateZPlay();
		player->SetPlayerVolume(audioVol,audioVol);
		player->OpenFile(f.str().c_str(),sfAutodetect);
		audios.push_back(player);
	}
	///////////////////准备工作结束

	//打开摄像机
	VideoCapture inputVideo(0);
	if(!inputVideo.isOpened())
	{
		//std::cout << "Open Failed" << std::endl;
		return -1;
	}

	int key = -1;
	Mat img_scene;
	cv::namedWindow("Cap",CV_WINDOW_NORMAL);
	
	cv::namedWindow("video",CV_WINDOW_NORMAL);
	cvSetWindowProperty( "video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );
	cvSetWindowProperty( "video", CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO );
	
	Mat EmptyMat = Mat::zeros(1,1,CV_8UC3);
	imshow("video",EmptyMat);

	boost::thread t1(play);
	
	while(key != 27)
	{
		////1-读取图像////
		inputVideo >> img_scene;
		imshow("Cap",img_scene);
		key = waitKey(10);
		if(key == 27)
		{
			//mutex_of_isEndOfApp.lock();
			isEndOfApp = true;
			//mutex_of_isEndOfApp.unlock();
		}

		Mat warp_dst = classifer.getCharacterImage(img_scene);
		if(warp_dst.empty())
		{
			//imshow("clip2",img_scene);
			continue;
		}
		if(_bug_flag)
		imshow("warp_dst",warp_dst);
		
/*		boost::format f("./sample-check/%d.bmp");
		f % index_of_remark;
		index_of_remark++;
		imwrite(f.str(),warp_dst);

		continue*/;

		pair<int,double> score = classifer.Score(warp_dst);
		if(_bug_flag)
			cout << score.first << endl;
		ids[index] = score.first/num_of_char_pose;
		if (ids[0] == ids[1] && ids[0] == ids[2] && ids[0]!=nowindex)
        {
			nowindex = ids[0];
			//mutex_of_nowPlay.lock();
			//last_play = now_play;
			now_play = (videos[nowindex]);
			//mutex_of_nowPlay.unlock();
			last_play_audio = now_play_audio;
			now_play_audio = audios[nowindex];
		}
		
        index++;
        index %= 3;
		//imshow("score",classifer.img_objects[score.first]);

		//boost::format f("./");
		//imwrite();
		
		//Mat clip = classifer.img_map_objects[score.first/num_of_char_pose];
		//imshow("clip2",clip);
		//imshow("clip2",classifer.AffineImage(img_scene,clip));
		//std::cout << score.first/4 << std::endl;
	}
	t1.join();
	return 0;
}
