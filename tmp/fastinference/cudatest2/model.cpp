#include <math.h>
#include <stdint.h>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "utils.h"

using namespace std;

namespace FAST_INFERENCE {

static constexpr signed char layer_2_weight[3][3][1][32] = {{{{1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1}}, {{1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1}}, {{1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1}}}, {{{1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1}}, {{-1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1}}, {{1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1}}}, {{{1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1}}, {{-1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1}}, {{1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1}}}};
static constexpr signed char layer_2_bias[32] = {-1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1};
static constexpr float layer_3_threshold[32] = {52972.41015625, 10309.9013671875, -48022.77734375, -30286.97265625, -29217.5, 10027.1904296875, -21053.53125, 20082.6015625, 48418.15625, -26492.30859375, 6108.40576171875, -46685.1171875, -30060.310546875, -49461.6640625, -7687.85498046875, -6599.97900390625, 75065.9609375, 80796.84375, -11970.6845703125, -10351.447265625, -31894.0546875, 10331.986328125, -10923.162109375, 24524.068359375, -7706.00146484375, -49909.2265625, 48404.42578125, -62.86895751953125, -33558.77734375, -67001.5625, 2661.91357421875, 4163.15185546875};
static constexpr unsigned int layer_5_weight[3][3][32][1] = {{{{0x88c9aebf}, {0x244babfb}, {0x63712936}, {0x5ded3728}, {0xd13be6d1}, {0x8c379204}, {0x24cc2860}, {0x2be2a701}, {0xafb5fea3}, {0x76a30646}, {0x2ccb3192}, {0xd805a936}, {0xfb1e2cc6}, {0x95a02722}, {0x106bea43}, {0x73de2bde}, {0x1369457a}, {0xfb0d26a}, {0x2a5e430f}, {0xb0c03527}, {0x586789}, {0xda467032}, {0xc6f48c22}, {0x3c17124c}, {0x3e4fac3}, {0x5d0f50fd}, {0x23c7678a}, {0x6b4e3a48}, {0xee4ca725}, {0xa9c0f134}, {0xc4ccd4ba}, {0xaa7e3e7b}}, {{0x7bdf3c5a}, {0xf8db4f7f}, {0x47b6c0f5}, {0x9564e6e2}, {0xb95412dc}, {0xfe5cc95d}, {0xcde96572}, {0xb374fa9b}, {0x7bdd6e57}, {0x4a83eb5d}, {0x3a2f2b10}, {0xc553e194}, {0x2db89a76}, {0xb214606c}, {0x187eaf8b}, {0x3bff7a6f}, {0x7abb2e7c}, {0xa7a9e6b0}, {0x227c0edb}, {0xe46c6507}, {0x83047d1a}, {0xc9474185}, {0xe4218262}, {0x5f4e7aef}, {0x9d2e599b}, {0xf12c4e5}, {0x462ecb14}, {0x33d5bc39}, {0xc4fa990}, {0x7909108}, {0xea33f23}, {0x7e7f0e6f}}, {{0x3e7f8a49}, {0x3a482f7f}, {0x693469b2}, {0x85c0152a}, {0x18440385}, {0x4440096}, {0xb0280812}, {0x7cd8e6a}, {0x32cf2acd}, {0x34ef6f4d}, {0xac4d89ce}, {0x91c3d3b4}, {0x7e2e0f78}, {0x14103210}, {0x1c6fa777}, {0xfaccab6d}, {0xbe6e1a8f}, {0x23f17a5a}, {0x83ecfdbb}, {0x4540cd66}, {0x60c187b9}, {0xa82376b4}, {0xc82cd827}, {0x8588d1e2}, {0xb4a189ed}, {0x19933a05}, {0xa8afee01}, {0xa7a7d6ae}, {0x482f2fd1}, {0x9af6de5c}, {0x3adf0d5d}, {0x7e2aeb7f}}}, {{{0x4dec10cb}, {0x62fc191f}, {0x5aed3e94}, {0x3b3918cc}, {0xfd3b3295}, {0xd9fffba1}, {0x3f5f18ff}, {0x6fcf5bfa}, {0x502ee151}, {0x2b5fcbf0}, {0x6977d7e1}, {0xdd4df1e7}, {0x304c07a8}, {0x31fda97c}, {0x5e3e3cdb}, {0x259b1eb}, {0xcf22352b}, {0x1b6fa78a}, {0x27bf571a}, {0xc43a2b14}, {0xe9c9e5ab}, {0x89805590}, {0x1e500846}, {0xb95e3e94}, {0x3d5a281a}, {0xfd557986}, {0x3f5f3ef1}, {0xcffbb64b}, {0x862367f5}, {0x86408010}, {0xa74cdc73}, {0x3dca1cce}}, {{0xabdb17dd}, {0xa6a1def}, {0xb9fdede4}, {0xd5eeb926}, {0x797da3f6}, {0xc8542d7c}, {0x3e9e9fdd}, {0xc501cc33}, {0x4459117b}, {0x7adeadcf}, {0x5b7e75e9}, {0x86cfc1aa}, {0x3bdc0088}, {0x2e9362ae}, {0xb8dcf54d}, {0xd81d5205}, {0x7baed7ef}, {0xe7cdfe2a}, {0x2f3f2e7d}, {0xa4032f11}, {0x46cf5998}, {0x5c832391}, {0x9d200234}, {0x7174fbcb}, {0x2227ca64}, {0x98134097}, {0xdece7b7f}, {0xb72e7d27}, {0xccfb67fb}, {0xa96380}, {0x45e9f763}, {0x75b71fed}}, {{0x70da13b1}, {0x20480fcf}, {0xf5b5d7bb}, {0x57a7ffa3}, {0x480b368e}, {0x4c145925}, {0xbb5f1abf}, {0x1741e5ab}, {0xc9f0d76b}, {0x3e7d3d7b}, {0xeabf1a54}, {0xceabeb94}, {0xfbff1e7f}, {0xd0904085}, {0x787a05a4}, {0x143506b7}, {0x24ee85e9}, {0xb3e9bd6a}, {0xd628cd5b}, {0x243faed6}, {0x1a23472b}, {0x1a59ab72}, {0x40c8d067}, {0xdd8bd527}, {0xaa0b2e70}, {0xf8556955}, {0xffeb355b}, {0xbfabfe6b}, {0x7abfa7c5}, {0x23e9bfb2}, {0x73eedd2b}, {0x7bfca67d}}}, {{{0x97b0d727}, {0x84808e3}, {0x721b1a65}, {0x7f4320df}, {0xe9b07280}, {0x345f0b3d}, {0x340a00df}, {0x6bcffbf2}, {0xc2aced61}, {0xb31f3d4d}, {0xf2ea56e1}, {0x584303f0}, {0x118574a}, {0x60d1657f}, {0x580d9740}, {0x88220ff2}, {0xf5b1bb12}, {0x8a777331}, {0x76f8d6c3}, {0x2a1d2e6e}, {0x40f33ee7}, {0x88035091}, {0x59942b5c}, {0x7d5c985c}, {0xefdc3cce}, {0xcc1049b4}, {0x1d70369d}, {0x7edf77e2}, {0x786faf7d}, {0x23a8956a}, {0x31054619}, {0x8fe02fda}}, {{0x7e4fe5b}, {0xd00e3d09}, {0x7443220a}, {0x1eac70d6}, {0x73f4953e}, {0x984e10d5}, {0x17de0ccb}, {0x24c01d97}, {0xc7e53dba}, {0x21d6b085}, {0x3740b2ce}, {0xba4aafbe}, {0x20a386f8}, {0xc1d0dd30}, {0xd9f7f6ec}, {0xf4a9acd0}, {0x79bcdf5b}, {0xf66f2dba}, {0xe33df10a}, {0xba7fbecb}, {0x4fcefd3f}, {0x581de5d8}, {0x182e4a56}, {0x576071d5}, {0x4bcf1eeb}, {0x589f6135}, {0x2f97f996}, {0xfffefaf8}, {0xbf5427fa}, {0xa3a98f38}, {0xaceb1e9e}, {0x3c3421df}}, {{0xc7a0c6e3}, {0x568e45a3}, {0x1e97bac8}, {0xffb5ce0d}, {0x2d7cd72a}, {0xe8d0d127}, {0x3a5a19ed}, {0x842b6b91}, {0x84b76fb1}, {0x31cdebbf}, {0xf9baf620}, {0x62cc140c}, {0x2bbb0678}, {0x4cc2c106}, {0xd968d66f}, {0x74815183}, {0x207faf10}, {0x97eb7ecb}, {0xa1ba5dc1}, {0x5add2ecd}, {0xc788056b}, {0x110709a2}, {0xb45ddadc}, {0x908eadb3}, {0x73ce28ce}, {0xd85c8b34}, {0xd6676dd6}, {0xf3f73eca}, {0xee18ace1}, {0x7ab8ff3}, {0x67ab1e9b}, {0x59dd3261}}}};
static constexpr signed char layer_5_bias[32] = {1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1};
static constexpr signed char layer_6_threshold[32] = {9, 10, 10, -1, 8, 2, 19, 3, 5, 30, 24, 6, 10, -24, 18, 14, 10, 15, 4, 14, -7, -5, -20, 13, 14, 8, 23, 38, 24, -16, 0, 30};
static constexpr unsigned int layer_9_weight[32][25] = {{0x4a4324c0, 0x6e6d65db, 0x6feefd3f, 0x9ffbe879, 0xaae3a93d, 0x8dc58060, 0xbde42848, 0x9cb38446, 0xb170ea3f, 0xb9aee077, 0x20104830, 0x31741cd4, 0x24b4c670, 0x3f150fb1, 0x1594de46, 0x339c23e4, 0xedb44c56, 0x27e7356a, 0x3dd36dfc, 0x7e7e45c9, 0x72103c9, 0x4d3b2660, 0xa4b325c6, 0xaeb6e8c7, 0xcb3803e0}, {0x89ac6b5, 0x27c7f792, 0x53bbfb3f, 0x9baee43b, 0x884a7501, 0xabe2908d, 0xe66bb78b, 0x72040521, 0x15844554, 0xacc60548, 0x4fe7c44b, 0xedeaf54b, 0xce13a184, 0x361417d0, 0x8edc1fc0, 0x3980a654, 0xa00836b8, 0x474f79f7, 0xacbfa968, 0x4bbf1b64, 0xb096c816, 0x4ae6d25c, 0x6e7b3301, 0x57122bfb, 0x529ee857}, {0x4e3077eb, 0x8890c7, 0xcdd54c8e, 0x374d807, 0x45401a80, 0xd9ae4961, 0x75f46d6a, 0xb390ed3e, 0x4318fc3f, 0x31c08a3e, 0x10140210, 0x89b1aeb7, 0x3180dc34, 0x530abc3b, 0xf201f69e, 0x324f59d1, 0x79b7857, 0xf26af8a9, 0x9b2163bf, 0xb8a7c140, 0x74c0d4c4, 0x81bdf937, 0xd9aff846, 0xf9fc49ff, 0x2eb496e1}, {0x69816571, 0xcc6fad60, 0xfdffccfb, 0xb7fed9eb, 0xd7559f33, 0x8d2d2928, 0x99840c6c, 0xa8f6e46a, 0xfb79b8bb, 0xfffbe8b7, 0x4172904, 0x11304a54, 0xa920c614, 0xf32129d9, 0xb389f715, 0x14104370, 0xb59c4816, 0x8554f857, 0xb52042fb, 0x2fb60759, 0x67373c0, 0x35595f03, 0xf96e33f, 0xffde68a7, 0xaa264260}, {0xd86f0342, 0x5c6ca509, 0x7c6c3358, 0x3669f7f9, 0x9bddbf76, 0x266012c0, 0x6f240a44, 0x8ced16c5, 0x7e6b13f1, 0x567bf209, 0xd76d7ebf, 0x6c518d41, 0x4e6e16d9, 0x2e2b23c1, 0xce6f6b41, 0xf9d7b12a, 0x235e6b, 0xcce7a041, 0xe6ca545, 0xd7597826, 0x3e7ab832, 0x6d865306, 0x541fbc, 0x45389fb1, 0x73dba9ba}, {0x3b5f529f, 0x83813846, 0x8b42b040, 0xc4003fa5, 0x98a40675, 0xb7b37bef, 0x178aabd3, 0x5559dbc7, 0x576bbab3, 0x66ddd733, 0xb3ecfdbf, 0x4745d7cb, 0xff6e30c9, 0x4faeebf5, 0x2c6eec41, 0xd983cadf, 0xcc87a767, 0xc8e7a559, 0x6e4bfccb, 0x1332fa27, 0xb7a8c78f, 0x67d15bab, 0x20341ed8, 0x60623ab2, 0xfd03e91b}, {0x50309295, 0x10d5300, 0x3025a00, 0x80e3e06, 0x622b1745, 0xd35db39a, 0x47eb1f81, 0x77013f04, 0x4c99e17, 0x50f16c1, 0xa767717d, 0xd6c7b569, 0x676a10c9, 0x460d2361, 0x2c0f00c1, 0x2efe80e5, 0xce6faf8f, 0xcee9a1c9, 0x6c2b2744, 0xd5880e04, 0xb1d9c887, 0x338c1b46, 0xdaec460b, 0x790b981b, 0x3743e93e}, {0x454341e1, 0xb8472651, 0x94801461, 0x1d11ab2, 0xa1d5decf, 0xc4f4471, 0x88840e68, 0x4a6a9c4a, 0x527abaab, 0x777af287, 0x3c176920, 0x9540e5c, 0x3b6894eb, 0xc76ba2bd, 0xfa0b2111, 0x2700c874, 0x993d4214, 0x8ae925e9, 0xea4f815d, 0x91546465, 0x2373d537, 0x31c8e8b7, 0x91ecf9ee, 0x57cddcae, 0x68b4dc62}, {0x682117e1, 0x4e2717c0, 0xcd2d39c0, 0x4a2620cf, 0x856a9b40, 0x4c5f2bc0, 0x1050d64, 0x1105ee16, 0xc092d61e, 0x339bea27, 0x2c1d6720, 0x1a094e5, 0x626012e9, 0x6403381, 0xab130340, 0x2a67fbe3, 0xea9ccbc7, 0xce6fa1c1, 0x8e0f0741, 0xc010540, 0x3f412df3, 0xe2192173, 0x4665424e, 0x4e2e1749, 0x661fb7e9}, {0xba6c2967, 0x7aeeefbb, 0x766e3758, 0xfe623a79, 0x978e693c, 0x77a9aaf, 0x63bd6b86, 0x8c6eb2c1, 0xbf0d8bf3, 0xb4954806, 0x93ffedbf, 0x53fddb96, 0xe6c56d1, 0xf2e4b71, 0x4fdf9ab8, 0xf9c5dd5a, 0xeb5e58ca, 0xe424b16c, 0x7dac2e95, 0xf7fb7a9f, 0x67065359, 0x79721048, 0xa83404cc, 0x39bf2450, 0xde459099}, {0x4181e03d, 0xb9bee831, 0xb6fbe3bb, 0xbf738f7b, 0xfdf84967, 0x32509488, 0x6be1f79, 0xaeff05cd, 0xdd7d0973, 0x76d15a3e, 0xeedfcdeb, 0x9b950332, 0xa199c61e, 0xaa6aecbc, 0xeb43f9b7, 0x5d4ced3b, 0x213e5199, 0x92086afe, 0xb32f39bd, 0xb9826115, 0x11d1f8b7, 0xadbcf816, 0x998de016, 0x9191e8de, 0x81f485cc}, {0xbfbcaa2f, 0x918dbb3e, 0xbe252d0, 0xa195be0, 0x50c5146, 0x2d13c096, 0x3b297aab, 0xc78eda87, 0xc21f9886, 0x724b3f0a, 0x8ac9f7b0, 0x6fc1b7cb, 0x77537f9b, 0x838ade1f, 0x7262a4bd, 0xca6ff9f9, 0x6e6bb2f9, 0xfa7720a1, 0x6a1e2052, 0x99240553, 0xc0efe93f, 0xe6eb31cb, 0xdfcd0b8e, 0xa4b5cef6, 0x6cee96e8}, {0xd1efe97d, 0x9d0c8194, 0xe012750, 0x4c0016cc, 0xcddb0750, 0x1494ffa7, 0x54213280, 0x3508ab56, 0x449c9744, 0x451d5fc2, 0x92406618, 0x4e4f1fe9, 0x76f232eb, 0xc805300, 0xcda74deb, 0xf637c6bc, 0xfee79ecd, 0xec67b749, 0x6e362f11, 0x59613cdf, 0xddeea5e9, 0x7ef7a760, 0xe4ee286e, 0xcc83a740, 0x77614e3}, {0x3f184376, 0x57946836, 0x47bcea26, 0x4c3ca006, 0x8226208d, 0x23e4c111, 0xd572e532, 0x7312e1be, 0x3300e4bd, 0xb100e314, 0x80a979b2, 0x56fcf8b7, 0x81a8c0be, 0xb0c7b89f, 0x4378d69b, 0xe0e8643f, 0xf7bbf137, 0x93c8fcbf, 0xb7134edf, 0xce2ec041, 0x28c6b9a, 0xd997e896, 0xd983e12b, 0x5b7c4e1f, 0xb10dc115}, {0xe8a10c68, 0xa8278e48, 0xbcee1d69, 0x5be37d5f, 0xd5f7b9b6, 0x44170321, 0x39a40862, 0x88a38c4d, 0x7b7bbabb, 0x73daea2f, 0x14050214, 0x39104a35, 0x3c14c651, 0x350933f7, 0xb509b390, 0x160c2fd0, 0x30968c48, 0x2c254748, 0x2c7417d0, 0x3b870540, 0x166311b8, 0x24302078, 0x3e25c918, 0x8cbf9541, 0x880817ec}, {0x9f4f2956, 0xb3edf9bf, 0x519ae7bc, 0x94066339, 0xf4ae6531, 0x878af036, 0x539dea3f, 0x269dc117, 0x31050f57, 0x8897cd58, 0xdd8cac0f, 0xc3c1cbdb, 0xa6f937c8, 0x3c3c1f20, 0x8594df42, 0x481436e8, 0xc4b7d00a, 0xc6ec4d42, 0x7de99d34, 0xd5dafbbe, 0x1b546a58, 0x39121488, 0x7f1a1478, 0x44705a7e, 0x976b3bab}, {0x35d7e8b7, 0x9196c278, 0x188c12d0, 0xb630af38, 0xb9f47450, 0x6128d6b1, 0xa3d46e3f, 0x4889c54f, 0x9e4ac920, 0x5779fc0a, 0xeb2e2c5b, 0x57bc8bde, 0xb1a0d52f, 0x3d54efbc, 0x61f2da3b, 0xf3ee322b, 0x37c6dd46, 0x95f0fe87, 0x73d9fbb7, 0xb9b4c835, 0x53ddf86, 0x35b4baac, 0x8175d637, 0xf965d35f, 0x2d4ecb4}, {0x710165ee, 0x54dd49d2, 0xfe5e2551, 0xcefd63eb, 0xcd1ec926, 0x64b6c356, 0x3c1ba012, 0x99a5a870, 0xd85c5116, 0xf59d8833, 0xa2b0ec36, 0x1208dfe9, 0x3e0d0a31, 0x88f4440e, 0xd1923dbe, 0x5ecbd7e8, 0xfee794c8, 0x34011f6c, 0x6f67439, 0x706334db, 0x4e7768c9, 0xacb68468, 0x8cb749e6, 0x86d52041, 0xee6c16e1}, {0xfed05d9f, 0x47b5f34, 0xc7bd7906, 0x802e6387, 0xc92fb24d, 0xd203df5d, 0x731d3b2f, 0x7350fb36, 0x131afe3e, 0x110dd69b, 0xcc7647ab, 0x2b97e906, 0x23c9d0bd, 0xd3206ab9, 0x9a06d49b, 0x88baa056, 0xab99f0b7, 0xa4b76bb, 0xef8f11b9, 0x88924600, 0x991cebb6, 0xcdc5fa5f, 0xd98f99cb, 0x7e5be866, 0x60df6b74}, {0x8f36c9b6, 0xff34a896, 0x3d058ce, 0x18a58c4, 0x4a6d16c4, 0xf11dcd37, 0xf3d9fc89, 0xf3d3cfae, 0xc687c826, 0x89ad244d, 0x9996de60, 0xdfefd2bb, 0xf3d35dab, 0xf193df36, 0xa5b4d817, 0x2d1f9492, 0x4e7bd2bd, 0xcbf322b1, 0x6f901977, 0x2eb443c3, 0xc8e884cd, 0xd6e970e1, 0x16cf1b89, 0x2d0d6eb, 0x4ee617c1}, {0xc0bf19e7, 0xc0443fc6, 0x19a1c86, 0x15c1886, 0x280132ec, 0xf57def46, 0x75fdf456, 0x93d2febf, 0xc9f8e9bb, 0xc82da0d7, 0xa370768c, 0x7602cbd, 0xb26f60e0, 0x73d7cd5e, 0xf198fe7e, 0xf86333c1, 0xd1978662, 0xf1908e7e, 0x81b1dcbe, 0xc7ffe9cd, 0x467426c0, 0x243016e8, 0xb310862c, 0x24507089, 0x330a2903}, {0xe4143a81, 0xc51f82, 0x810819a6, 0x41001b04, 0x22096c9, 0x9b562592, 0xcbd5fb26, 0x5380df3e, 0x4398fa9e, 0x834460f1, 0x31b51f5d, 0x3927feff, 0xa2e255aa, 0x4392fd3f, 0x8f85c287, 0xf82f857a, 0x7d96e9d7, 0x8af28641, 0xff8a6977, 0xe85d3a1, 0x2c94eb36, 0xa090dfa8, 0xd185eb16, 0x125ec91b, 0x701fa831}, {0x77bd0a32, 0x3c91a91e, 0xb60092d8, 0x165017a2, 0x46991ec2, 0x31b3137a, 0x91de1e25, 0x8df688a6, 0x464abcab, 0x467b3aaf, 0xdfefe3eb, 0xdf850373, 0x5973dd6e, 0xe9ebf83e, 0xe36b213f, 0x5679e0df, 0x4f623195, 0x1a7a2001, 0xfa2f6131, 0x9181c610, 0xb3fcb8fb, 0xc7e8b9bf, 0x83899295, 0xe4a14ef7, 0x2ce85f4a}, {0xd38fc413, 0xf7a7cd9b, 0xd765d78f, 0xcece28ce, 0xd7fe4116, 0x2914a931, 0xf88aea76, 0x516f27f6, 0xcc7708c5, 0x14481f8e, 0x31364680, 0xfb80df9a, 0xe7c33eae, 0x81a952c6, 0xea6afecd, 0xd2f8de97, 0x3e78f3b1, 0x774f73b3, 0x10497a61, 0xaa9d04d1, 0xfce99cff, 0xceebb1c9, 0x7e6f1bc1, 0x6ee737e0, 0xd871668}, {0x6afbd2e9, 0x7a180b61, 0x788b5530, 0xf5e7d43b, 0xa8ef7338, 0xde041e63, 0x98754554, 0x2a7a04f9, 0x2e6020f9, 0xbc4457c0, 0x5d142ce6, 0x88783480, 0x88242054, 0xbc3d23d4, 0x349957d0, 0x80910bc8, 0xd08cac60, 0x8595cd9e, 0xce1adeeb, 0x56162b28, 0x34137382, 0x9113233e, 0x2c56c485, 0xc6332689, 0x9149e834}, {0xc2e3cae, 0x921f0aaa, 0x75222b24, 0x382a2215, 0x2a6fa4e9, 0xe094c417, 0xd0b87b35, 0x3104eb36, 0x2084d314, 0x8995e755, 0x91c0f538, 0x56ef7b6d, 0x70f345db, 0x3e1a39a0, 0x31945da6, 0xe86e0548, 0x5de60c68, 0xa0218e49, 0x1fdaadfc, 0xd6db6dbf, 0xd262130, 0x39360f34, 0xb912ec9c, 0xdf74cc1f, 0x760973db}, {0x9e26b4da, 0xdaeef6f9, 0xbec735f9, 0x5e9a10eb, 0xf7ec55fe, 0x357f69ff, 0x9fae8aff, 0xec7c141, 0xf5951fdb, 0xecff8b5f, 0xa5870068, 0x29d50f58, 0x8e7a17c9, 0x3c5f07f0, 0xd197c916, 0x2e160260, 0x25f7dedc, 0xcce79540, 0x37cb9ab4, 0x35dcdff6, 0x4e2b3441, 0x64c2675, 0x1c2c1b5d, 0x8e90b7e3, 0x6e6e97e9}, {0xbeb39365, 0xc5d47ab6, 0x81b47d86, 0xc125f98f, 0x89242aa4, 0xaab2d13e, 0x7311eb36, 0xd311f636, 0xf1bdfab4, 0x9b98f4b4, 0x826a7d1b, 0x136c79b2, 0x938ce8a6, 0xd1a5dc1e, 0xd7f0dcef, 0xc7eee47f, 0xf7ebe9b7, 0x31e9fe36, 0xb1f5e63f, 0x446d3988, 0xdbe4da7f, 0xd9d31dfe, 0xc7ffac69, 0x8f362160, 0x9d520804}, {0xeff97f4a, 0x472f0b3e, 0x76603b06, 0xf9263337, 0x8ebee107, 0x199be21e, 0x54013a96, 0x3339fb17, 0xd21dda16, 0xf1954c26, 0xeb7dfeab, 0x60b1ba9, 0x226a1a9b, 0x62e2b7b, 0x177afa9f, 0xf9a38569, 0xcfff0c4b, 0xe8efa4e9, 0xb66b261d, 0x26636bdf, 0x2a1487a6, 0x31361e26, 0xd9aecc77, 0xdfa8e15c, 0xcd9d1900}, {0x29ee1a26, 0x12003e79, 0xb857788d, 0x8e8bb392, 0x5a498645, 0xb06affd9, 0xc7cb7eff, 0xf5779759, 0x4061703, 0xcac0649, 0xd78fc26d, 0xebc5fd4f, 0xcef715e9, 0x75150561, 0x1c944544, 0x30159972, 0xac121608, 0x4cde3941, 0x45908f6e, 0xc6fcdeae, 0xe0178ca9, 0x26692048, 0x667b13f8, 0x7279deab, 0x7ffbf9db}, {0x827a3300, 0xc0095d80, 0x89d15ef6, 0x2098d367, 0x5869fe8b, 0xbf435828, 0x24209466, 0x11311c38, 0x4ee375af, 0x4f6223ff, 0x27790497, 0xa789cb8, 0xca8f2964, 0xc6f1c5bf, 0x29001760, 0xc62714c0, 0xf1852d6e, 0x3194cc5e, 0x48f2f46a, 0xc66a3039, 0x305b1b81, 0x30300e34, 0x31905ca6, 0x93581988, 0xd19468bf}, {0x100f178e, 0xcce3b86e, 0xceae61e9, 0xc76efacd, 0xdf7e69d6, 0x442f1720, 0x9f9eaaea, 0xc9faea57, 0x7b2ff2b3, 0xe2cbf4b3, 0xc066606, 0x3185c46, 0x1e64328b, 0x9f8f6846, 0xda6af459, 0xf6f0ffb7, 0xc2e9f7bf, 0xf78fe313, 0x61919656, 0x2c3e7efa, 0xe66937ff, 0xee4b85e9, 0x467b3ec9, 0x6efe5361, 0x90027180}};
static constexpr signed char layer_9_bias[32] = {-1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1};
static constexpr signed char layer_10_threshold[32] = {57, -25, 14, 34, -14, -12, -71, -29, 1, 12, 4, 2, -24, 3, -21, -12, 5, 0, 20, 16, 11, -9, -14, 19, -40, 11, 41, 47, 3, 17, -17, 59};
static constexpr unsigned int layer_11_weight[10][1] = {{0xfbe7a4ec}, {0x4aa2d586}, {0x6e458ccf}, {0x6234b718}, {0x5863d1a5}, {0x3fb6a719}, {0x3f339ee6}, {0x66db9d3d}, {0xf2a7acda}, {0xf086b595}};
static constexpr signed char layer_11_bias[10] = {-1, 1, 1, 1, 1, 1, -1, -1, 1, 1};

static double cuda_layer_2_output[1*26*26*32];
static double layer_2_output[1][26][26][32];

static unsigned int cuda_layer_3_output[1*26*26*1*32];
static unsigned int layer_3_output[1][26][26][1];

static unsigned int cuda_layer_4_output[1*13*13*1*32];
static unsigned int layer_4_output[1][13][13][1];

static signed int cuda_layer_5_output[1*11*11*32];
static signed int layer_5_output[1][11][11][32];

static unsigned int cuda_layer_6_output[1*11*11*1*32];
static unsigned int layer_6_output[1][11][11][1];

static unsigned int cuda_layer_7_output[1*5*5*1*32];
static unsigned int layer_7_output[1][5][5][1];

static signed int cuda_layer_9_output[1*32];
static signed int layer_9_output[1][32];

static unsigned int cuda_layer_10_output[1*1*32];
static unsigned int layer_10_output[1][1];

static signed int cuda_layer_11_output[1*10];
static signed int layer_11_output[1][10];



void predict_cudatest2(double const * const x, double * pred) {
    auto layer_0_output = x;
    double sum_cpu = 0, sum_gpu = 0;

	
    // Layer 1: Reshape
    auto layer_1_output = (double (*)[28][1]) layer_0_output;

    // Layer 2: Conv
    for(int b = 0; b < 1; b++){
    	for (int h = 0; h < 26; h++) {
    		for (int w = 0; w < 26; w++) {
    			for (int m = 0; m < 32; m++) {
    				//layer_2_output[b][h][w][m] = layer_2_bias[m];
     				cuda_layer_2_output[index4D(b,h,w,m,26,26,32)] = layer_2_bias[m];
    			}
    			for (int kH = 0; kH < 3; kH++) {
    				for (int kW = 0; kW < 3; kW++) {
    					for (int c = 0; c < 1; c++) {
    						for (int m = 0; m < 32; m++) {
                  // not sure if [b] needed in layer_1_output[?][h * 1 + kH - 0][w * 1 + kW - 0][c];
    							//layer_2_output[b][h][w][m] += layer_2_weight[kH][kW][c][m] * layer_1_output[h * 1 + kH - 0][w * 1 + kW - 0][c];
    							cuda_layer_2_output[index4D(b,h,w,m,26,26,32)] += layer_2_weight[kH][kW][c][m] * layer_1_output[h * 1 + kH - 0][w * 1 + kW - 0][c];
    						}
    					}
    				}
    			}
    		}
    	}
    }

    // // checksum L2 = 
    // ofstream g2("layer2/orig.out");
    // for(int b = 0; b < 1; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 26; h++) {// 
    //     for (int w = 0; w < 26; w++) {
    //       for (int m = 0; m < 32; m++) {
    //         sum_cpu += layer_2_output[b][h][w][m];
    //         g2<<layer_2_output[b][h][w][m]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 2(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    // // checksum L2 = 
    // ofstream gg2("layer2/par.out");
    // for(int b = 0; b < 1; b++){
    //   sum_gpu = 0;
    //   for(int i=b*26*26*32;i<(b+1)*26*26*32;i++){
    //     sum_gpu += cuda_layer_2_output[i];
    //     gg2<<cuda_layer_2_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 2(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 3: Step
    for(int b = 0; b < 1; b++){
      for (int h = 0; h < 26; h++) {
        for (int w = 0; w < 26; w++) {
          for (int c = 0; c < 32; c++) {
            // if (layer_2_output[b][h][w][c] >= layer_3_threshold[c]) {
            if (cuda_layer_2_output[index4D(b,h,w,c,26,26,32)] >= layer_3_threshold[c]) {  
              layer_3_output[b][h][w][c / 32] |= (1U << (31 - c % 32));
            } else {
              layer_3_output[b][h][w][c / 32] &= ~(1U << (31 - c % 32));
            }
          }
        }
      }
    }

    // // might be needed, but subsequent layers work with normal layer_x_output from step
    // for(int b = 0; b < 1; b++){
    //   for (int h = 0; h < 26; h++) {
    //     for (int w = 0; w < 26; w++) {
    //       for (int c = 0; c < 32; c++) {
    //         cuda_layer_3_output[index4D(b,h,w,c,26,26,32)] = layer_3_output[b][h][w][c];
    //       }
    //     }
    //   }
    // }

    // 
    // Layer 4: MaxPool
    for(int b = 0; b < 1; b++){
      for (int h = 0; h < 13; h++) {
        for (int w = 0; w < 13; w++) {
          for (int c = 0; c < 1; c++) {
            //layer_4_output[b][h][w][c] = 0;
            cuda_layer_4_output[index4D(b,h,w,c,13,13,1)] = 0;
          }
          for (int kH = 0; kH < 2; kH++) {
            for (int kW = 0; kW < 2; kW++) {
              for (int c = 0; c < 1; c++) {
                //layer_4_output[b][h][w][c] |= layer_3_output[b][h * 2 + kH][w * 2 + kW][c];
                cuda_layer_4_output[index4D(b,h,w,c,13,13,1)] |= layer_3_output[b][h * 2 + kH][w * 2 + kW][c];
              }
            }
          }
        }
      }
    }

    // // checksum L4 = 
    // ofstream g4("layer4/orig.out");
    // for(int b = 0; b < 1; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 13; h++) {// 
    //     for (int w = 0; w < 13; w++) {
    //       for (int c = 0; c < 1; c++) {
    //         sum_cpu += layer_4_output[b][h][w][c];
    //         g4<<layer_4_output[b][h][w][c]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 4(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    // // checksum L4 = 
    // ofstream gg4("layer4/par.out");
    // for(int b = 0; b < 1; b++){
    //   sum_gpu = 0;
    //   for(int i=b*13*13*32;i<(b+1)*13*13*32;i++){
    //     sum_gpu += cuda_layer_4_output[i];
    //     gg4<<cuda_layer_4_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 4(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 5: Conv
    for(int b = 0; b < 1; b++){
      for (int h = 0; h < 11; h++) {
        for (int w = 0; w < 11; w++) {
          for (int m = 0; m < 32; m++) {
            //layer_5_output[b][h][w][m] = layer_5_bias[m];
            cuda_layer_5_output[index4D(b,h,w,m,11,11,32)] = layer_5_bias[m];
          }
          for (int kH = 0; kH < 3; kH++) {
            for (int kW = 0; kW < 3; kW++) {
              for (int m = 0; m < 32; m++) {
                for (int c = 0; c < 1; c++) {
                  //layer_5_output[b][h][w][m] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_5_weight[kH][kW][m][c] ^ cuda_layer_4_output[index4D(b,(h * 1 + kH - 0),(w * 1 + kW - 0),c,13,13,1)])) - 32;
                  cuda_layer_5_output[index4D(b,h,w,m,11,11,32)] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_5_weight[kH][kW][m][c] ^ cuda_layer_4_output[index4D(b,(h * 1 + kH - 0),(w * 1 + kW - 0),c,13,13,1)])) - 32;
                }
              }
            }
          }
        }
      }
    }

    // // checksum L5 = 
    // ofstream g5("layer5/orig.out");
    // for(int b = 0; b < 1; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 11; h++) {// 
    //     for (int w = 0; w < 11; w++) {
    //       for (int m = 0; m < 32; m++) {
    //         sum_cpu += layer_5_output[b][h][w][m];
    //         g5<<layer_5_output[b][h][w][m]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 5(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    // // checksum L5 = 
    // ofstream gg5("layer5/par.out");
    // for(int b = 0; b < 1; b++){
    //   sum_gpu = 0;
    //   for(int i=b*11*11*32;i<(b+1)*11*11*32;i++){
    //     sum_gpu += cuda_layer_5_output[i];
    //     gg5<<cuda_layer_5_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 5(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 6: Step
    for(int b = 0; b < 1; b++){
      for (int h = 0; h < 11; h++) {
        for (int w = 0; w < 11; w++) {
          for (int c = 0; c < 32; c++) {
            // if (layer_5_output[b][h][w][c] >= layer_6_threshold[c]) {
            if (cuda_layer_5_output[index4D(b,h,w,c,11,11,32)] >= layer_6_threshold[c]) {  
              layer_6_output[b][h][w][c / 32] |= (1U << (31 - c % 32));
            } else {
              layer_6_output[b][h][w][c / 32] &= ~(1U << (31 - c % 32));
            }
          }
        }
      }
    }

    // // might be needed, but subsequent layers work with normal layer_x_output from step
    // for(int b = 0; b < 1; b++){
    //   for (int h = 0; h < 11; h++) {
    //     for (int w = 0; w < 11; w++) {
    //       for (int c = 0; c < 32; c++) {
    //         cuda_layer_6_output[index4D(b,h,w,c,11,11,32)] = layer_6_output[b][h][w][c];
    //       }
    //     }
    //   }
    // }

    // 
    // Layer 7: MaxPool
    for(int b = 0; b < 1; b++){
      for (int h = 0; h < 5; h++) {
        for (int w = 0; w < 5; w++) {
          for (int c = 0; c < 1; c++) {
            //layer_7_output[b][h][w][c] = 0;
            cuda_layer_7_output[index4D(b,h,w,c,5,5,1)] = 0;
          }
          for (int kH = 0; kH < 2; kH++) {
            for (int kW = 0; kW < 2; kW++) {
              for (int c = 0; c < 1; c++) {
                //layer_7_output[b][h][w][c] |= layer_6_output[b][h * 2 + kH][w * 2 + kW][c];
                cuda_layer_7_output[index4D(b,h,w,c,5,5,1)] |= layer_6_output[b][h * 2 + kH][w * 2 + kW][c];
              }
            }
          }
        }
      }
    }

    // // checksum L7 = 
    // ofstream g7("layer7/orig.out");
    // for(int b = 0; b < 1; b++){
    //   sum_cpu = 0;
    //   for (int h = 0; h < 5; h++) {// 
    //     for (int w = 0; w < 5; w++) {
    //       for (int c = 0; c < 1; c++) {
    //         sum_cpu += layer_7_output[b][h][w][c];
    //         g7<<layer_7_output[b][h][w][c]<<" ";  
    //       }
    //     }
    //   }
    //   cout<<fixed<<"layer 7(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;

    // // checksum L7 = 
    // ofstream gg7("layer7/par.out");
    // for(int b = 0; b < 1; b++){
    //   sum_gpu = 0;
    //   for(int i=b*5*5*32;i<(b+1)*5*5*32;i++){
    //     sum_gpu += cuda_layer_7_output[i];
    //     gg7<<cuda_layer_7_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 7(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // Layer 8: Reshape
    // auto layer_8_output = (unsigned int (*)) layer_7_output;
    auto cuda_layer_8_output = (unsigned int (*)) cuda_layer_7_output;

    // Layer 9: Gemm
    for(int b = 0; b < 1; b++){
      for (int d = 0; d < 32; d++) {
        // layer_9_output[b][d] = layer_9_bias[d];
        cuda_layer_9_output[b*32 + d] = layer_9_bias[d];
      }
      for (int d = 0; d < 32; d++) {
        for (int i = 0; i < 25; i++) {
          // layer_9_output[b][d] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_9_weight[d][i] ^ cuda_layer_8_output[i])) - 32;
          cuda_layer_9_output[b*32 + d] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_9_weight[d][i] ^ cuda_layer_8_output[b*25 + i])) - 32;
        }
      }
    }

    // // checksum L9
    // ofstream gg9("layer9/par.out");
    // for(int b = 0; b < 1; b++){
    //   sum_gpu = 0;
    //   for(int i=b*32;i<(b+1)*32;i++){
    //     sum_gpu += cuda_layer_9_output[i];
    //     gg9<<cuda_layer_9_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 9(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // // checksum L9 
    // ofstream g9("layer9/orig.out");
    // for(int b = 0; b < 1; b++){
    //   sum_cpu = 0;
    //   for (int d = 0; d < 32; d++) {
    //     sum_cpu += layer_9_output[b][d];
    //     g9<<layer_9_output[b][d]<<" ";  
    //   }
    //   cout<<fixed<<"layer 9(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;
    // 
    // Layer 10: Step
    for(int b = 0; b < 1; b++){
      for (int d = 0; d < 32; d++) {
        if (cuda_layer_9_output[b*32 + d] >= layer_10_threshold[d]) {
          layer_10_output[b][d / 32] |= (1U << (31 - d % 32));
        } else {
          layer_10_output[b][d / 32] &= ~(1U << (31 - d % 32));
        }
      }
    }

    signed int *cuda_layer_10_output = (signed int *) layer_10_output;

    // Layer 11: Gemm
    for(int b = 0; b < 1; b++){
      for (int d = 0; d < 10; d++) {
        //layer_11_output[b][d] = layer_11_bias[d];
        cuda_layer_11_output[b*10 + d] = layer_11_bias[d];
      }
      for (int d = 0; d < 10; d++) {
        for (int i = 0; i < 1; i++) {
          //layer_11_output[b][d] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_11_weight[d][i] ^ cuda_layer_10_output[b*1 + i])) - 32;
          cuda_layer_11_output[b*10 + d] += 2 * __builtin_popcount((unsigned int)~(unsigned int)(layer_11_weight[d][i] ^ cuda_layer_10_output[b*1 + i])) - 32;
        }
      }
    }

    // // checksum L11
    // ofstream gg11("layer11/par.out");
    // for(int b = 0; b < 1; b++){
    //   sum_gpu = 0;
    //   for(int i=b*10;i<(b+1)*10;i++){
    //     sum_gpu += cuda_layer_11_output[i];
    //     gg11<<cuda_layer_11_output[i]<<" ";  
    //   }
    //   cout<<fixed<<"layer 11(GPU): batch "<<b<<": "<<sum_gpu<<endl;
    // }
    // cout<<endl;

    // // checksum L11 
    // ofstream g11("layer11/orig.out");
    // for(int b = 0; b < 1; b++){
    //   sum_cpu = 0;
    //   for (int d = 0; d < 10; d++) {
    //     sum_cpu += layer_11_output[b][d];
    //     g11<<layer_11_output[b][d]<<" ";  
    //   }
    //   cout<<fixed<<"layer 11(CPU): batch "<<b<<": "<<sum_cpu<<endl;
    // }
    // cout<<endl;
    // 

    for(int b = 0; b < 1; b++){
      for (int i = 0; i < 10; i++) {
        pred[i] += layer_11_output[b][i];
      }
    }
}

}