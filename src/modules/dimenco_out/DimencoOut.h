/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DimencoOut.h
 * Author: user
 *
 * Created on 12. dubna 2016, 13:50
 */

#ifndef DIMENCOOUT_H
#define DIMENCOOUT_H

#include "yuri/core/thread/SpecializedIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/event/BasicEventConsumer.h"

namespace yuri {
    namespace dimencoout {

        class DimencoOut : public core::SpecializedIOFilter<core::RawVideoFrame> {
            using base_type = core::SpecializedIOFilter<core::RawVideoFrame>;
        public:
            IOTHREAD_GENERATOR_DECLARATION
            static core::Parameters configure();
            DimencoOut(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
            virtual ~DimencoOut() noexcept;
        private:

            virtual core::pFrame do_special_single_step(core::pRawVideoFrame frame) override;
            virtual bool set_param(const core::Parameter& param) override;
        };

    } /* namespace mosaic */
} /* namespace yuri */

#endif /* DIMENCOOUT_H */

