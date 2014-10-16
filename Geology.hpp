#pragma once
#ifndef _GEOLOGY_H_
#define _GEOLOGY_H_
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

///œﬂ∂Œœ‡Ωª
Point2d CrossPoint(Point2d p1,Point2d p2,Point2d p3, Point2d p4)
{	
	if(p1.x == p2.x)
	{
		Point2d pt1(p1.x , p3.y + (p1.x - p3.x)/(p4.x - p3.x)*(p4.y - p3.y));
		return pt1;
	}

	if(p3.x == p4.x)
	{
		Point2d pt1(p3.x , p1.y + (p3.x - p1.x)/(p2.x - p1.x)*(p2.y - p1.y));
		return pt1;
	}
	if(p1.y == p2.y)
	{
		Point2d pt1(p3.x + (p1.y - p3.y)/(p4.y - p3.y)*(p4.x - p3.x) , p1.y);
		return pt1;
	}

	if(p3.y == p4.y)
	{
		Point2d pt1(p1.x + (p3.y - p1.y)/(p2.y - p1.y)*(p2.x - p1.x) , p3.y);
		return pt1;
	}

	double r = ((p1.y - p3.y)*(p4.x - p3.x)-(p1.x - p3.x)*(p4.y - p3.y))/((p2.x - p1.x)*(p4.y - p3.y)-(p2.y - p1.y)*(p4.x - p3.x));
	Point2d pt1 = p1 + r*(p2 - p1);
	return pt1;

}


bool isSegmentCross(Point2d p1,Point2d p2,Point2d p3, Point2d p4)
{
	double xmin1 , xmax1 ,ymin1 , ymax1;
	double xmin2 , xmax2 ,ymin2 , ymax2;
	xmin1 = std::min(p1.x,p2.x);
	xmin2 = std::min(p3.x,p4.x);
	xmax1 = std::max(p1.x,p2.x);
	xmax2 = std::max(p3.x,p4.x);
	ymin1 = std::min(p1.y,p2.y);
	ymin2 = std::min(p3.y,p4.y);
	ymax1 = std::max(p1.y,p2.y);
	ymax2 = std::max(p3.y,p4.y);

	//cout << p1 << p2 << p3 << p4 ;
	if(xmax1<xmin2 || xmax2<xmin1 || ymax1<ymin2 || ymax2<ymin1)
	{
		//cout << "out !" << endl;
		return false;
	}

	//Point2d pt1 = p1 - p3;
	//Point2d pt2 = p4 - p3;
	//Point2d pt3 = p2 - p3;

	//double ktmp1 = pt1.ddot(pt2);
	//double ktmp2 = pt2.ddot(pt3);
	//double k = pt1.ddot(pt2) * pt2.ddot(pt3);
	//
	////cout << "in !" << k << endl; 
	//return (k > 0);
	Point2d CP = CrossPoint(p1,p2,p3,p4);
	return (CP.x >= xmin1 && CP.x >= xmin2 && CP.x <= xmax1 && CP.x <= xmax2 && CP.y >= ymin1 && CP.y >= ymin2 && CP.y <= ymax1 && CP.y <= ymax2 );
}

template<typename T>
bool isPointInTriangle(T P , T A ,T B , T C)
{
	T v0 = C - A;
	T v1 = B - A;
	T v2 = P - A;
	double dot00 = v0.dot(v0);
	double dot01 = v0.dot(v1);
	double dot02 = v0.dot(v2);
	double dot11 = v1.dot(v1);
	double dot12 = v1.dot(v2);

	double inverDeno = 1/(dot00 * dot11 - dot01 * dot01);
	double u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
	if(u < 0 || u >1)
	{
		return false;
	}

	double v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
	if(v < 0 || v >1)
	{
		return false;
	}

	return u+v <= 1;
}

///////
double det(Point p0,Point p1,Point p2)  
{  
    return (p1.x-p0.x)*(p2.y-p0.y)-(p1.y-p0.y)*(p2.x-p0.x);  
}  
  
double ploygon_area(vector<Point> p)    
{  
	int n = p.size();
    double s=0.0f;  
    int i=1;  
    for(;i < n-1;i++)  
	{
        s += det(p[0],p[i],p[i+1]);  
	}
    return 0.5*fabs(s);  
}  
#endif