/*!
 * @file 		Anaglyph.h
 * @author 		Zdenek Travnicek
 * @date 		31.9.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#ifndef ANAGLYPH_H_
#define ANAGLYPH_H_

namespace yuri {

namespace anaglyph {
using anaglyph_base = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
class Anaglyph: public anaglyph_base
{
public:
	/// Standard constructor
	/// @param _log  logger
	/// @param parent  parent thread
	/// @param correction Correction in pixels meaning how many pixels to the right should be right image shifted
									Anaglyph(log::Log &_log,
					core::pwThreadBase parent, const core::Parameters& parameters);
	virtual 						~Anaglyph() noexcept;
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters 		configure();
protected:
	//virtual bool step();
	std::vector<core::pFrame>	do_special_step(const std::tuple<core::pRawVideoFrame, core::pRawVideoFrame>& frames);
	bool 							set_param(const core::Parameter& param);
protected:
	int correction;
//	bool fast;
};


}

}

#endif /* ANAGLYPH_H_ */
