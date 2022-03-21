Forward Error Correction (FEC)
==============================

The Forward Error Correction (FEC) package provides encoding and decoding
algorithms for several coding schemes such as low-density parity-check (LDPC),
Polar, and convolutional codes as well as cyclic redundancy checks (CRC).

Although most codes are 5G compliant, the decoding
algorithms are mostly general and can be used in combination with other
code designs.

Besides the encoding/decoding algorithms, this package also provides
interleavers, scramblers, and rate-matching for seamless integration of the FEC
package into the remaining physical layer processing chain.

.. toctree::
   :maxdepth: 3

   fec.ldpc
   fec.polar
   fec.conv
   fec.crc
   fec.interleaving
   fec.scrambling
   fec.utils
