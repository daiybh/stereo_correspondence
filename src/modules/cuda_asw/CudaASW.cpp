/*!
 * @file 		CudaASW.cpp
 * @author 		Your name <lhotamir@fit.cvut.cz>
 * @date 		10.03.2016
 * @copyright	Institute of Intermedia, CTU in Prague, 2016
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "CudaASW.h"
#include "yuri/core/Module.h"
#include "asw.h"
#include "yuri/core/frame/raw_frame_types.h"


namespace yuri {
    namespace cuda_asw {

        IOTHREAD_GENERATOR(CudaASW)

        MODULE_REGISTRATION_BEGIN("cuda_asw")
        REGISTER_IOTHREAD("cuda_asw", CudaASW)
        MODULE_REGISTRATION_END()

        core::Parameters CudaASW::configure() {
            core::Parameters p = core::IOThread::configure();
            p.set_description("CudaASW");
            p["num_disparities"]["Number of disparities"]=16;
            p["iterations"]["Number of refinement iterations"]=6;
            return p;
        }

        CudaASW::CudaASW(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters) :
        core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 2, std::string("cuda_asw")),num_disparities(16),iterations(6) {
            IOTHREAD_INIT(parameters)
            supported_formats_.push_back(core::raw_format::y8);
        }

        CudaASW::~CudaASW() noexcept {
        }

        std::vector<core::pFrame> CudaASW::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) {
            converter_left = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            converter_right = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            core::pRawVideoFrame left_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_left->convert_to_cheapest(std::get<0>(frames), supported_formats_));
            core::pRawVideoFrame right_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_right->convert_to_cheapest(std::get<1>(frames), supported_formats_));
            size_t w = left_frame->get_width();
            size_t h = left_frame->get_height();
            const unsigned char *left_data = PLANE_RAW_DATA(left_frame, 0);
            const unsigned char *right_data = PLANE_RAW_DATA(right_frame, 0);
            int* d = disparity(left_data, right_data, num_disparities, w, h, iterations);
            unsigned char* out = new unsigned char[w * h];
            int coef = 256 / num_disparities;
            for (unsigned int i = 0; i < (w * h); i++) {
                out[i] = d[i] * coef;
            }
            core::pRawVideoFrame output = core::RawVideoFrame::create_empty(core::raw_format::y8,{static_cast<dimension_t> (w), static_cast<dimension_t> (h)},
            out,w * h * sizeof (unsigned char));
            return{output, left_frame};
        }

        bool CudaASW::set_param(const core::Parameter& param) {
            if (assign_parameters(param)
                    (num_disparities, "num_disparities")
                    (iterations, "iterations"))
                return true;
            return core::IOThread::set_param(param);
        }

    } /* namespace cuda_asw */
} /* namespace yuri */
