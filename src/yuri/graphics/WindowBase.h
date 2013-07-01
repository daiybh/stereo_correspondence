/*!
 * @file 		WindowBase.h
 * @author 		Zdenek Travnicek
 * @date 		31.5.2008
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2008 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef WINDOWBASE_H_
#define WINDOWBASE_H_

#include "yuri/log/Log.h"
#include "yuri/core/ThreadBase.h"
#include "yuri/core/Parameters.h"

namespace yuri
{
namespace graphics
{

class WindowBase: public yuri::core::ThreadBase
{
public:
	WindowBase(log::Log &log_, core::pwThreadBase parent, core::Parameters &p);
	virtual ~WindowBase();
	virtual void run() {}
	virtual inline int get_width() { return width; }
	virtual inline int get_height() { return height; }
	virtual inline int get_x() { return x; }
	virtual inline int get_y() { return y; }
	virtual bool resize(unsigned int, unsigned int) {return false;}
	virtual void show(bool /*value*/=true) {};
	virtual bool create()=0;
	virtual void swap_buffers() {}
	virtual bool check_key(int /*keysym*/) { return false; }
	virtual void exec(core::pCallback) { }
	virtual bool have_stereo() { return false; }
	virtual inline std::string getName() { return name; }
	virtual bool process_events() { return false; }
protected:
	yuri::ssize_t x,y;
	yuri::size_t width,height;
	std::map<int,bool> keys;
	yuri::mutex keys_lock;
	std::string name;
	core::pCallback keyCallback;
	core::Parameters params;
};

}
}
#endif /*WINDOWBASE_H_*/
