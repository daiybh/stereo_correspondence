/*!
 * @file 		BasicFrame.cpp
 * @author 		Zdenek Travnicek
 * @date 		28.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "BasicFrame.h"
#include <boost/foreach.hpp>
#include "yuri/exception/OutOfRange.h"

namespace yuri {

namespace core {

BasicFrame::BasicFrame(yuri::size_t planes):
		/*numberPlanes(0),*/dts(0),pts(0),duration(0),format(YURI_FMT_NONE),
		width(0),height(0),samples(1),channels(1)
{
	set_planes_count(planes);
}

BasicFrame::~BasicFrame() {
}

plane_t &BasicFrame::get_plane(yuri::size_t index)
{
	if (index >= planes.size())
		throw exception::OutOfRange();
	return planes[index];
}

plane_t &BasicFrame::operator[](yuri::size_t index)
{
	return get_plane(index);
}
yuri::size_t BasicFrame::get_size()
{
	yuri::size_t size = 0;
	for (std::vector<plane_t>::iterator it=planes.begin();it!=planes.end();++it) {
		size += it->size();
	}
	return size;
}

void BasicFrame::set_planes_count(yuri::size_t count)
{
	planes.resize(count);
}

yuri::size_t BasicFrame::get_planes_count()
{
	return planes.size();
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
void BasicFrame::set_plane(yuri::size_t index, const plane_t& plane)
{
	if (index >= planes.size()) throw exception::OutOfRange("Plane number out of range");
	planes[index] = plane;
}
void BasicFrame::set_plane(yuri::size_t index, plane_t& plane)
{
	if (index >= planes.size()) throw exception::OutOfRange("Plane number out of range");
	planes[index].swap(plane);
}
void BasicFrame::set_plane(yuri::size_t index, const yuri::ubyte_t *data, yuri::size_t data_size)
{
	if (index >= planes.size()) throw exception::OutOfRange("Plane number out of range");
	plane_t &plane = planes[index];
	//plane.clear();
	plane.reserve(data_size);
//	std::copy(data,data+data_size,std::back_inserter(plane));
	plane.insert(plane.begin(),data,data+data_size);
}
pBasicFrame BasicFrame::get_copy()
{
	pBasicFrame tmp ( new BasicFrame(planes.size()));
	tmp->set_parameters(format, width, height);
	tmp->set_time(pts,dts,duration);
	tmp->set_info(info);
	for (yuri::size_t i = 0; i < planes.size(); ++i) {
		tmp->get_plane(i)=const_cast<const plane_t&>(get_plane(i));
	}
	return tmp;
}

}


}
