/*
 * BasicFrame.cpp
 *
 *  Created on: Jul 28, 2010
 *      Author: neneko
 */

#include "BasicFrame.h"
#include <boost/foreach.hpp>

namespace yuri {

namespace io {

using namespace yuri::exception;
BasicFrame::BasicFrame(yuri::size_t planes):
		numberPlanes(0),dts(0),pts(0),duration(0),format(YURI_FMT_NONE),
		width(0),height(0),samples(1),channels(1)
{
	set_planes_count(planes);
}

BasicFrame::~BasicFrame() {
	//if(info) delete info;
}

Plane<yuri::ubyte_t> &BasicFrame::operator[](yuri::size_t index)
{
	if (index >= numberPlanes)
		throw OutOfRange();
	shared_ptr<Plane<yuri::ubyte_t> > p(planes[index]);
	if (!p.get()) {
		p = shared_ptr<Plane<yuri::ubyte_t> > (new Plane<yuri::ubyte_t>());
		planes[index] = p;
	}
	return *p;
}
yuri::size_t BasicFrame::get_size()
{
	shared_ptr<Plane<yuri::ubyte_t> > plane;
	yuri::size_t size = 0;
	BOOST_FOREACH(plane, planes) {
		size += plane->size;
	}
	return size;
}

void BasicFrame::set_planes_count(yuri::size_t count)
{
	planes.resize(count);
	numberPlanes = count;
}

yuri::size_t BasicFrame::get_planes_count()
{
	return numberPlanes;
}

void BasicFrame::set_parameters(yuri::format_t format, yuri::size_t width, yuri::size_t height, yuri::size_t channels, yuri::size_t samples)
{
	this->format = format;
	this->width = width;
	this->height = height;
	this->channels=channels;
	this->samples=samples;
}
void BasicFrame::set_time(yuri::size_t pts, size_t dts, size_t duration)
{
	this->pts = pts;
	this->dts = dts;
	this->duration = duration;
}

yuri::size_t BasicFrame::get_width()
{
	return width;
}

yuri::size_t BasicFrame::get_height()
{
	return height;
}
yuri::format_t BasicFrame::get_format()
{
	return format;
}
yuri::usize_t BasicFrame::get_sample_count()
{
	return samples;
}
yuri::usize_t BasicFrame::get_channel_count()
{
	return channels;
}
void BasicFrame::set_plane(yuri::size_t index, shared_ptr<Plane<yuri::ubyte_t> > plane)
{
	if (index >= numberPlanes) throw OutOfRange("Plane number out of range");
	planes[index] = plane;
}

pBasicFrame BasicFrame::get_copy()
{
	pBasicFrame tmp ( new BasicFrame(numberPlanes));
	tmp->set_parameters(format, width, height);
	tmp->set_time(pts,dts,duration);
	for (yuri::size_t i = 0; i < numberPlanes; ++i) {
		shared_ptr<Plane<yuri::ubyte_t> > p = planes[i]->get_copy();
		tmp->set_plane(i,p);
	}
	return tmp;
}

}


}
