/*!
 * @file 		Anaglyph.h
 * @author 		Zdenek Travnicek
 * @date 		31.9.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include <yuri/io/BasicIOThread.h>
#include <yuri/config/Config.h>
#include "yuri/config/RegisteredClass.h"

#ifndef ANAGLYPH_H_
#define ANAGLYPH_H_

namespace yuri {

namespace io {
using yuri::log::Log;
using namespace yuri::config;
using namespace std;
class Anaglyph: public BasicIOThread {
public:
	PACK_START
	struct _rgb {
		char r,g,b;
	} PACK_END;
	PACK_START struct _rgba {
			char r,g,b,a;
	} PACK_END;
	/// Standard constructor
	/// @param _log  logger
	/// @param parent  parent thread
	/// @param correction Correction in pixels meaning how many pixels to the right should be right image shifted
	Anaglyph(Log &_log, pThreadBase parent, int correction = 0, bool fast = true);
	virtual ~Anaglyph();
	static shared_ptr<BasicIOThread> generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception);
	static shared_ptr<Parameters> configure();
protected:
	virtual bool step();
	template<typename T> shared_ptr<BasicFrame> makeAnaglyph(shared_ptr<BasicFrame> left, shared_ptr<BasicFrame> right);
protected:
	int correction;
	bool fast;
};


}

}

#endif /* ANAGLYPH_H_ */
