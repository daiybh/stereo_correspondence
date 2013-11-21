/*!
 * @file 		Anaglyph.h
 * @author 		Zdenek Travnicek
 * @date 		31.9.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include <yuri/core/BasicIOFilter.h>

#ifndef ANAGLYPH_H_
#define ANAGLYPH_H_

namespace yuri {

namespace anaglyph {

class Anaglyph: public core::BasicMultiIOFilter {
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
									Anaglyph(log::Log &_log,
					core::pwThreadBase parent, core::Parameters& parameters);
	virtual 						~Anaglyph();
	IO_THREAD_GENERATOR_DECLARATION
	static core::pParameters 		configure();
protected:
	//virtual bool step();
	std::vector<core::pBasicFrame>	do_single_step(const std::vector<core::pBasicFrame>& frames);
	template<typename T> core::pBasicFrame
									makeAnaglyph(const core::pBasicFrame& left,
					const core::pBasicFrame& right);

	bool 							set_param(const core::Parameter& param);
protected:
	int correction;
//	bool fast;
};


}

}

#endif /* ANAGLYPH_H_ */
