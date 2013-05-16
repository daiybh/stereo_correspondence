/*
 * ReadPcap.cpp
 *
 *  Created on: 26.2.2013
 *      Author: neneko
 */

#include "ReadPcap.h"
#include "yuri/core/Module.h"
#include <iomanip>
#include <arpa/inet.h>

namespace yuri {
namespace pcap {

REGISTER("read_pcap",ReadPcap)
IO_THREAD_GENERATOR(ReadPcap)

core::pParameters ReadPcap::configure()
{
	core::pParameters p = core::BasicIOThread::configure();
	p->set_description("ReadPcap module.");
	(*p)["filename"]["File to read from"]=std::string();
	p->set_max_pipes(1,1);
	return p;
}
namespace {
struct pcap_header_ {
	uint32_t 	magic;
	struct {
		uint16_t major;
		uint16_t minor;
	} version;
	int32_t 	zone;
	uint32_t	sigfigs;
	uint32_t	snaplen;
	uint32_t	network;
};
struct packet_header_ {
	uint32_t 	sec;
	uint32_t 	usec;
	uint32_t 	len;
	uint32_t 	orig_len;
};
struct ip_header_ {
	uint8_t		mac_dest[6];
	uint8_t		mac_src[6];
	uint16_t	type;
};
struct ipv4_header_ {
	uint8_t		version;//:4;
//	uint8_t		header_length:4;
	uint8_t		dsf;
	uint16_t	total_length;
	uint16_t	id;
	uint16_t	flags;
	uint8_t		ttl;
	uint8_t		protocol;
	uint16_t	checksum;
	uint32_t	src_ip;
	uint32_t	dst_ip;
};
struct udp_header_ {
	uint16_t	src_port;
	uint16_t	dst_port;
	uint16_t	length;
	uint16_t	checksum;
};
}


ReadPcap::ReadPcap(log::Log &log_, core::pwThreadBase parent, core::Parameters &parameters):
core::BasicIOThread(log_,parent,1,1,std::string("ReadPcap")),packet_count(0),last_valid_size(0)
{
	IO_THREAD_INIT("ReadPcap")
	file_.open(filename_.c_str(),std::ios::in|std::ios::binary);
	if (!file_.is_open()) throw exception::InitializationFailed("Failed to open open file");
	pcap_header_ header;
	file_.read(reinterpret_cast<char*>(&header),sizeof(pcap_header_));
	if (!file_.good()) throw exception::InitializationFailed("Failed to read pcap file header");
	log[log::info] << "Opened pcap file with version " << header.version.major << "."
			<< header.version.major << ", with capture of network " << header.network <<"\n";
	if (header.network != 1) throw exception::InitializationFailed("Only ethernet captures supported now");

}

ReadPcap::~ReadPcap()
{
}

void ReadPcap::run()
{
	IO_THREAD_PRE_RUN
	packet_header_ pkt_header;
	std::vector<char> data;
	while(still_running()) {
		file_.read(reinterpret_cast<char*>(&pkt_header),sizeof(packet_header_));
		if(!file_.good()) break;
		if(pkt_header.len>65536) {
			log[log::warning] << "Read packet with an invalid length " << pkt_header.len << ". Nothing to read anymore\n";
			if (last_valid_size) pkt_header.len =  last_valid_size;
			else {
				log[log::error] << "No previous valid packet to guess size from, bailing out\n";
				break;
			}
//			break;
		} else last_valid_size = pkt_header.len;
		log[log::debug] << "Read packet from time " << pkt_header.sec << "." <<std::setfill('0') << std::setw(6)  << pkt_header.usec
				<< ". Captured " << pkt_header.len << "/" << pkt_header.orig_len << " bytes\n";

		data.resize(pkt_header.len);
		file_.read(&data[0],pkt_header.len);
		if(!file_.good()) break;
		packet_count++;
		if (data.size() < sizeof(ip_header_)) {
			log[log::warning] << "Packet too small to parse\n";
			continue;
		}
		ip_header_ *ip_hdr = reinterpret_cast<ip_header_*>(&data[0]);
		if (ip_hdr->type != 0x0008) {
			log[log::warning] << "Packet is not an IPv4 packet\n";
			continue;
		}
		if (data.size() < sizeof(ip_header_)+sizeof(ipv4_header_)) {
			log[log::warning] << "Packet too small to parse\n";
			continue;
		}
		ipv4_header_ *ipv4_hdr = reinterpret_cast<ipv4_header_*>(&data[sizeof(ip_header_)]);
		if (ipv4_hdr->protocol != 17) {
			log[log::warning] << "Packet is not an UDP packet\n";
			continue;
		}
		uint8_t header_length = ipv4_hdr->version&0x0F;
		if (header_length != 5) {
			log[log::warning] << "Unsupported header size ("<<static_cast<int>(header_length)<<")\n";
			continue;
		}
		if (data.size() < sizeof(ip_header_)+sizeof(ipv4_header_)+sizeof(udp_header_)) {
			log[log::warning] << "Packet too small to parse\n";
			continue;
		}
		udp_header_* udp_header = reinterpret_cast<udp_header_*>(&data[sizeof(ip_header_)+sizeof(ipv4_header_)]);
		uint16_t data_len = ntohs(udp_header->length) - sizeof(udp_header_);
		log[log::debug] << "Found UDP packet with " << data_len << " bytes of data\n";
		core::pBasicFrame frame = allocate_frame_from_memory(reinterpret_cast<ubyte_t*>(&data[sizeof(ip_header_)+sizeof(ipv4_header_)+sizeof(udp_header_)]),data_len);
		push_video_frame(0,frame,YURI_FMT_NONE,0,0);

	}
	log[log::info] << "Read total " << packet_count << " packets\n";
	IO_THREAD_POST_RUN
	request_end(YURI_EXIT_OK);
	/*if (!in[0]) return true;
	core::pBasicFrame frame = in[0]->pop_frame();
	if (!frame) return true;

	if(!discard) push_raw_frame(0,frame);
	return true;*/
}
bool ReadPcap::set_param(const core::Parameter &param)
{
	if (param.name == "filename") {
		filename_ = param.get<std::string>();
	} else return core::BasicIOThread::set_param(param);
	return true;
}

} /* namespace pcap */
} /* namespace yuri */
