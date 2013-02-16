/*!
 * @file 		TSMuxer.cpp
 * @author 		Zdenek Travnicek
 * @date 		10.8.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "TSMuxer.h"
namespace yuri {

namespace video {


REGISTER("tsmuxer",TSMuxer)


shared_ptr<BasicIOThread> TSMuxer::generate(Log &_log,pThreadBase parent,Parameters& /*parameters*/)
	throw (Exception)
{
	shared_ptr<TSMuxer> s(new TSMuxer(_log,parent));
	return s;
}
shared_ptr<Parameters> TSMuxer::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	p->set_description("MPEG2 TS Muxer based on libavformat");
	//(*p)["format"]["Color format for the output"]="RGB";
	p->set_max_pipes(1,1);

	p->add_input_format(YURI_VIDEO_MPEG2);
	p->add_output_format(YURI_VIDEO_MPEGTS);
	p->add_converter(YURI_VIDEO_MPEG2,YURI_VIDEO_MPEGTS,0,true);
	return p;
}


TSMuxer::TSMuxer(Log &_log, pThreadBase parent):AVCodecBase(_log,parent,"TSMux"),
		output_format(0),buffer_size(188*70),buffer_position(0)
{
	av_register_all();
	latency=10000;
}

TSMuxer::~TSMuxer() {

}

bool TSMuxer::step()
{
	if (!in[0] || in[0]->is_empty()) return true;
	if (in[0]->get_type() != YURI_TYPE_VIDEO) {
		log[warning] << "Input pipe is not video, bailing out" << std::endl;
		return true;
	}
	yuri::size_t pkts = 0;
	ptime t1 = microsec_clock::local_time();
	shared_ptr<BasicFrame> frame;
	while ((frame=in[0]->pop_frame())) {
		if (reconfigure(frame)) {
			if (!put_frame(frame)) return false;
		}
		pkts++;
	}
	ptime t2 = microsec_clock::local_time();
	log[debug] << "Processed " << pkts << " input frames in " <<
			to_simple_string(t2-t1) << "s (" << to_simple_string(t1.time_of_day())
			<< " - " << to_simple_string(t2.time_of_day()) << std::endl;
	return true;
}

bool TSMuxer::reconfigure(shared_ptr<BasicFrame> frame)
{
	// sanity checks
	assert (frame);
	if (!output_format) {
		output_format = av_guess_format("mpegts",0,0);
		if (!output_format) {
			log[error] << "Failed to get format for mpeg ts" << std::endl;
			return false;
		}
	}
	if (!format_context) {
		format_context.reset(avformat_alloc_context(),AVCodecBase::av_deleter<AVFormatContext>);
		if (!format_context) {
			log[error] << "Failed to allocate format context." << std::endl;
			return false;
		}
		format_context->oformat = output_format;
	}
	if (frame->get_format() != YURI_VIDEO_MPEG2) {
		log[warning] << "Input format is not MPEG video. (Got " <<
				BasicPipe::get_format_string(frame->get_format()) << ")";
		return false;
	}
	if (!format_context->nb_streams) {
		shared_ptr<AVStream> s(avformat_new_stream(format_context.get(),0));
		s->id = 0;
		streams.push_back(s);
		s->codec->codec_id = CODEC_ID_MPEG2VIDEO;
		s->codec->codec_type = AVMEDIA_TYPE_VIDEO;
		s->codec->bit_rate = 400000;
		s->codec->width = frame->get_width();
		s->codec->height = frame->get_height();
		s->codec->time_base.den = 25;
		s->codec->time_base.num = 1;
		s->codec->gop_size = 12;
		s->codec->pix_fmt = PIX_FMT_YUV420P;
		if(format_context->oformat->flags & AVFMT_GLOBALHEADER)
			s->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
		/*if (av_set_parameters(format_context.get(),0)<0) {
			log[error] << "Failed to set parameters for context" << std::endl;
			return false;
		}*/
		//byte_context.reset(new AVIOContext);
		//if (!buffer) buffer = allocate_memory_block(buffer_size+FF_INPUT_BUFFER_PADDING_SIZE);
		buffer.resize(buffer_size+FF_INPUT_BUFFER_PADDING_SIZE);
		//init_put_byte(byte_context.get(),(uint8_t *)buffer.get(),buffer_size,1,
		//    (void*)this, TSMuxer::read_buffer,TSMuxer::write_buffer,TSMuxer::seek_buffer);
		byte_context.reset(avio_alloc_context((uint8_t *)&buffer[0],buffer_size,1,
				(void*)this, TSMuxer::read_buffer,TSMuxer::write_buffer,TSMuxer::seek_buffer));
		format_context->pb = byte_context.get();
		//av_write_header(format_context.get());
		avformat_write_header(format_context.get(),0);
	}
	return true;

}

int TSMuxer::read_buffer(void *opaque, yuri::ubyte_t* buf, int size)
{
	if (!opaque) return -1;
	TSMuxer * muxer = static_cast<TSMuxer*>(opaque);
	return muxer->_read_buffer(buf,size);
}
int TSMuxer::write_buffer(void *opaque, yuri::ubyte_t* buf, int size)
{
	if (!opaque) return -1;
	TSMuxer * muxer = static_cast<TSMuxer*>(opaque);
	return muxer->_write_buffer(buf,size);
}
int64_t TSMuxer::seek_buffer(void* opaque, int64_t offset, int whence)
{
	if (!opaque) return -1;
	TSMuxer * muxer = static_cast<TSMuxer*>(opaque);
	return muxer->_seek_buffer(offset, whence);
}

int TSMuxer::_read_buffer(yuri::ubyte_t* /*buf*/, int /*size*/)
{
	log[error] << "TSMuxer::_read_buffer() not implemented" << std::endl;
	return -1;

	//return memcpy(buf, buffer.get(), size);
}
int TSMuxer::_write_buffer(yuri::ubyte_t* buf, int size)
{
	log[debug] << "Reading " << size << " bytes (188x" << size/188 << " + "
			<< size%188 <<  ")"<< std::endl;
	if (out[0]) {
		shared_ptr<BasicFrame> frame = allocate_frame_from_memory(buf,size);
		push_video_frame(0,frame,YURI_VIDEO_MPEGTS,0,0);
	}
	return size;
}
int64_t TSMuxer::_seek_buffer(int64_t offset, int whence)
{
	switch (whence) {
		case SEEK_SET: buffer_position = offset;
		case SEEK_CUR: buffer_position += offset;
		default: return -1;
	}
	log[debug] << "Seeked " << offset << " bytes. Actual position " << buffer_position <<std::endl;
	return buffer_position;
}

bool TSMuxer::put_frame(shared_ptr<BasicFrame> frame)
{
	shared_ptr<AVPacket> packet(new AVPacket,AVCodecBase::av_deleter<AVPacket>);
	av_init_packet(packet.get());
	packet->data = (uint8_t*)PLANE_RAW_DATA(frame,0);
	packet->size = PLANE_SIZE(frame,0);
	packet->stream_index = 0;
	yuri::size_t time_step = 1e6 * format_context->streams[0]->time_base.num / format_context->streams[0]->time_base.den;
	packet->pts = frame->get_pts() / time_step;
	packet->dts = frame->get_dts() / time_step;
	packet->duration = frame->get_duration() / time_step;

	av_write_frame(format_context.get(),packet.get());
	return true;
}

}

}

