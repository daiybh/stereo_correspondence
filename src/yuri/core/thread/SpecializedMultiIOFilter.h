/*
 * SpecializedMultiIOFilter.h
 *
 *  Created on: 2.10.2013
 *      Author: neneko
 */

#ifndef SPECIALIZEDMULTIIOFILTER_H_
#define SPECIALIZEDMULTIIOFILTER_H_
#include "MultiIOFilter.h"
#include <tuple>
#include <cassert>
namespace yuri {
namespace core {


template<typename Iter>
std::tuple<> verify_types(Iter)
{
	return {};
}

template<class FrameType, class... Others, typename Iter>
std::tuple<shared_ptr<FrameType>, shared_ptr<Others>...>
verify_types(Iter frame_iter)
{
	shared_ptr<FrameType> frame = dynamic_pointer_cast<FrameType>(*frame_iter);
	if (!frame) throw std::runtime_error("Wrong type");
	return std::tuple_cat(std::make_tuple(frame), verify_types<Others...>(frame_iter+1));
}



template<class... InFrameTypes>
class SpecializedMultiIOFilter: public MultiIOFilter {
public:
	static const size_t input_frames_count = sizeof...(InFrameTypes);
	using param_type = std::tuple<shared_ptr<InFrameTypes>... >;

	SpecializedMultiIOFilter(const log::Log &log_, pwThreadBase parent, position_t out_p,
			const std::string& id = "Spec. Multi Filter")
	:MultiIOFilter(log_, parent, input_frames_count, out_p, id) {}

	~SpecializedMultiIOFilter() noexcept {}
private:
	virtual std::vector<pFrame> do_single_step(const std::vector<pFrame>& frames) override final {
		assert (frames.size() == input_frames_count);
		try {
//			Timer t;
			const auto& p = verify_types<InFrameTypes...>(frames.begin());
//			auto t1 = t.get_duration();
			const auto& f = do_special_step(p);
//			auto t2 = t.get_duration();
//			log[log::info] << "Processing took "<<t2<<" in total, conversions " << t1 << " and module took " << (t2-t1);
			return f;
//			return do_special_step();
		}
		catch (std::exception& e) {
			log[log::info] << "Wrong frame types specified: " << e.what();
			throw;
		}
	}

	virtual std::vector<pFrame> do_special_step(const param_type& frames) = 0;

};


}
}

#endif /* SPECIALIZEDMULTIIOFILTER_H_ */
