/*
 * DECOMPRESSOR: Streaming Decompression Engine for Layer 7 (Huffman + RLE)
 * Purpose: On-the-fly decompression of chunks during transmission to clients
 * 
 * Architecture:
 *   - Pipelined Huffman decoder (entropy decoding)
 *   - RLE (Run-Length Encoding) expander
 *   - Sliding window support for inter-chunk dependencies
 *   - Zero-copy streaming: compressed block in → original payload out
 * 
 * Throughput Target:
 *   - Sustain decompression at 200 Gbps (25 GB/s input compressed)
 *   - Output: ~12.5 TB/s (at 500x compression ratio)
 * 
 * Pipeline Stages:
 *   Stage 0: Input buffering from NVMe/network
 *   Stage 1: Block header parsing
 *   Stage 2: Huffman symbol decode (pipelined)
 *   Stage 3: RLE expand
 *   Stage 4: Output formatting + checksum
 * 
 * Strategy:
 *   - Huffman tables cached in on-chip BRAM (multiple canonical tables)
 *   - Each 4 MiB chunk has own Huffman table header (4 KB)
 *   - Hardware decode: 2 symbols per cycle from Huffman tree
 *   - RLE fully pipelined: detect repeat symbols, emit run count
 * 
 * Implementation Target: Xilinx UltraScale+
 * Frequency: 250 MHz (4 ns cycle)
 */

module decompressor #(
    parameter INPUT_WIDTH = 512,            // Compressed input (64 bytes per cycle)
    parameter OUTPUT_WIDTH = 512,           // Decompressed output (64 bytes per cycle)
    parameter MAX_SYMBOLS = 256,            // Huffman alphabet size (0-255 bytes)
    parameter CHUNK_SIZE = 4194304,         // 4 MiB chunks
    parameter HUFF_TABLE_SIZE = 4096        // Huffman table per chunk (4 KB)
)(
    // Clock and reset
    input logic clk,
    input logic rst_n,
    
    // --- COMPRESSED INPUT PATH ---
    input logic [INPUT_WIDTH-1:0] comp_data_in,
    input logic [6:0] comp_data_valid,      // Validity mask (7 x 64-bit words)
    input logic [11:0] comp_data_len,       // Compressed bytes in word (0-64)
    input logic comp_block_start,           // Start of new 4 MiB block
    input logic comp_block_last,            // End of block
    output logic comp_data_ready,
    
    // --- HUFFMAN TABLE CONFIG (write-only) ---
    input logic huff_cfg_valid,
    input logic [11:0] huff_cfg_addr,       // Table entry address
    input logic [15:0] huff_cfg_code_len,   // Huffman code length bits
    input logic [15:0] huff_cfg_code_val,   // Encoded value
    input logic [7:0] huff_cfg_symbol,      // Output symbol (0-255)
    output logic huff_cfg_rdy,
    
    // --- DECOMPRESSED OUTPUT PATH ---
    output logic [OUTPUT_WIDTH-1:0] decomp_data_out,
    output logic [6:0] decomp_data_valid,
    output logic [11:0] decomp_data_len,
    output logic decomp_block_valid,
    input logic decomp_ready,
    
    // --- CRC/CHECKSUM (for integrity) ---
    output logic [31:0] crc32_out,
    output logic crc32_valid
);

    // ==========================================
    // STAGE 0: INPUT BUFFERING & BIT EXTRACTION
    // ==========================================
    // Compressed data is bitstream; need to extract individual bits for Huffman decoder
    
    localparam INPUT_FIFO_DEPTH = 256;
    logic [INPUT_WIDTH-1:0] fifo_data [0:INPUT_FIFO_DEPTH-1];
    logic [11:0] fifo_len [0:INPUT_FIFO_DEPTH-1];
    logic [8:0] fifo_head, fifo_tail, fifo_count;
    
    // FIFO write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_head <= 0;
            fifo_count <= 0;
        end else if (comp_data_valid != 0) begin
            fifo_data[fifo_head] <= comp_data_in;
            fifo_len[fifo_head] <= comp_data_len;
            fifo_head <= (fifo_head + 1) % INPUT_FIFO_DEPTH;
            fifo_count <= fifo_count + 1;
        end
    end
    
    assign comp_data_ready = (fifo_count < (INPUT_FIFO_DEPTH - 16));  // Maintain headroom
    
    // ==========================================
    // BIT-LEVEL EXTRACTION (convert byte stream to bit stream)
    // ==========================================
    
    logic [63:0] bit_buffer;        // Sliding bit buffer
    logic [7:0] bit_pos;            // Current position in buffer (0-63)
    logic [15:0] bits_available;    // Bits left in FIFO
    


    logic [DATA_WIDTH-1:0] fifo_peek;
    assign fifo_peek = fifo_data[fifo_tail];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_buffer <= 0;
            bit_pos <= 0;
            bits_available <= 0;
        end else begin
            // Track bit availability
            if (fifo_count > 0) begin
                bits_available <= (fifo_len[fifo_tail] * 8) + (64 - bit_pos);
            end else begin
                bits_available <= 0;
            end
            
            // Refill bit_buffer when below 32 bits
            if (bit_pos > 32 && fifo_count > 0) begin
                bit_buffer <= fifo_peek[63:0];  // Load next 64 bits
                bit_pos <= 0;
                fifo_tail <= (fifo_tail + 1) % INPUT_FIFO_DEPTH;
                fifo_count <= fifo_count - 1;
            end
        end
    end
    
    // ==========================================
    // STAGE 1: HUFFMAN TABLE & DECODER
    // ==========================================
    // Canonical Huffman decoding: use sorted code-length array to decode
    
    // Huffman table storage: BRAM-based for fast lookup
    // Table structure: [code_length | symbol_value]
    logic [23:0] huff_table [0:HUFF_TABLE_SIZE-1];  // (16-bit code_len | 8-bit symbol)
    
    // Configuration path
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            huff_cfg_rdy <= 1;
        end else if (huff_cfg_valid) begin
            huff_table[huff_cfg_addr] <= {huff_cfg_code_len, huff_cfg_symbol};
            huff_cfg_rdy <= 1;
        end
    end
    
    // Huffman decoder: extract bits and match against table
    // Two-stage: prefix match + symbol lookup
    
    logic [15:0] huff_code;        // Current Huffman code being decoded
    logic [4:0] huff_code_len;     // Bits in current code
    logic [7:0] huff_symbol_out;
    logic huff_match_valid;
    
    // Simple shift-based decoder (proof-of-concept)
    // In production: use memoization or first-fit search
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            huff_code <= 0;
            huff_code_len <= 0;
            huff_symbol_out <= 0;
            huff_match_valid <= 0;
        end else if (bits_available >= 16) begin
            // Extract next 16 bits
            huff_code <= bit_buffer[bit_pos +: 16];
            
            // Search table for matching code
            for (int i = 0; i < HUFF_TABLE_SIZE; i++) begin
                logic [15:0] tbl_code_len;
                logic [7:0] tbl_symbol;
                
                {tbl_code_len, tbl_symbol} = huff_table[i];
                
                // Check if code matches (first tbl_code_len bits)
                if (tbl_code_len > 0 && tbl_code_len <= 16) begin
                    if ({bit_buffer[bit_pos +: tbl_code_len], {(16-tbl_code_len){1'b0}}} ==
                        {huff_code[15:(16-tbl_code_len)], {(16-tbl_code_len){1'b0}}}) begin
                        huff_symbol_out <= tbl_symbol;
                        huff_code_len <= tbl_code_len[4:0];
                        huff_match_valid <= 1;
                        break;
                    end
                end
            end
            
            if (huff_match_valid) begin
                bit_pos <= (bit_pos + huff_code_len) % 64;
            end
        end
    end
    
    // ==========================================
    // STAGE 2: RLE (Run-Length Encoding) EXPANDER
    // ==========================================
    // Input: stream of Huffman symbols
    // Output: expanded runs (e.g., symbol=255 + count=10 → 10 copies of previous symbol)
    
    logic [7:0] prev_symbol, curr_symbol;
    logic [15:0] run_count;
    logic rle_emit;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prev_symbol <= 0;
            run_count <= 0;
            rle_emit <= 0;
        end else if (huff_match_valid) begin
            // Check for RLE marker (e.g., symbol 255 signals run header)
            if (huff_symbol_out == 8'hFF) begin
                // Next symbol is run count (low byte = count, high byte in next symbol)
                run_count <= {huff_symbol_out, 8'b0};  // Shift for count accumulation
                rle_emit <= 0;
            end else if (run_count > 0) begin
                // Emit run_count copies of this symbol
                curr_symbol <= huff_symbol_out;
                rle_emit <= 1;
                run_count <= run_count - 1;
            end else begin
                curr_symbol <= huff_symbol_out;
                rle_emit <= 1;
                prev_symbol <= huff_symbol_out;
            end
        end
    end
    
    // ==========================================
    // STAGE 3: OUTPUT ASSEMBLY
    // ==========================================
    // Pack decompressed symbols into 512-bit output words
    
    logic [OUTPUT_WIDTH-1:0] output_buf;
    logic [5:0] output_pos;        // Symbols packed so far (0-64)
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_buf <= 0;
            output_pos <= 0;
            decomp_data_valid <= 0;
        end else if (rle_emit && decomp_ready) begin
            // Pack symbol into output buffer
            output_buf[output_pos * 8 +: 8] <= curr_symbol;
            output_pos <= output_pos + 1;
            
            // Emit full word when packed
            if (output_pos == 63) begin
                decomp_data_out <= output_buf;
                decomp_data_valid <= 7'b1111111;  // All 7 words valid
                decomp_data_len <= 64;
                output_pos <= 0;
            end
        end else if (comp_block_last && output_pos > 0) begin
            // Flush partial word at end of block
            decomp_data_out <= output_buf;
            decomp_data_valid <= {{(64-output_pos){1'b0}}, {output_pos{1'b1}}};
            decomp_data_len <= output_pos;
            decomp_block_valid <= 1;
            output_pos <= 0;
        end
    end
    
    // ==========================================
    // STAGE 4: CRC32 CHECKSUM
    // ==========================================
    // Compute running CRC32 of decompressed output for integrity
    
    logic [31:0] crc32_state;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            crc32_state <= 32'hFFFFFFFF;
            crc32_valid <= 0;
        end else if (rle_emit) begin
            // CRC32-CCITT polynomial: update for new symbol
            crc32_state <= crc32_poly_update(crc32_state, curr_symbol);
        end else if (decomp_block_last) begin
            crc32_out <= ~crc32_state;  // Invert final state
            crc32_valid <= 1;
        end
    end
    
    // ==========================================
    // CRC32 POLYNOMIAL COMPUTATION (inline)
    // ==========================================
    function logic [31:0] crc32_poly_update (input logic [31:0] crc, input logic [7:0] byte_in);
        logic [31:0] temp;
        temp = crc ^ {24'b0, byte_in};
        for (int i = 0; i < 8; i++) begin
            if (temp[0]) begin
                temp = (temp >> 1) ^ 32'hEDB88320;  // CRC32-IEEE polynomial
            end else begin
                temp = temp >> 1;
            end
        end
        crc32_poly_update = temp ^ crc[31:8] ^ 24'b0;  // Simplified
    endfunction

endmodule


// ============================================================================
// DECOMPRESSOR_TOP: Integration module with buffer management
// ============================================================================

module decompressor_top #(
    parameter NUM_CLIENTS = 64              // Max concurrent client streams
)(
    input logic clk,
    input logic rst_n,
    
    // --- INPUT from NVMe-oF (compressed chunks) ---
    input logic [511:0] nvme_data,
    input logic [63:0] nvme_valid,
    input logic [11:0] nvme_len,
    output logic nvme_ready,
    
    // --- OUTPUT to network (decompressed, per-client) ---
    output logic [NUM_CLIENTS-1:0] [511:0] client_data,
    output logic [NUM_CLIENTS-1:0] client_valid,
    input logic [NUM_CLIENTS-1:0] client_ready,
    
    // --- Chunk routing (which client gets this chunk) ---
    input logic [5:0] chunk_client,         // Client ID (0-63)
    input logic chunk_route_valid
);

    // Generate NUM_CLIENTS decompressor instances, each with separate buffering
    generate
        for (genvar c = 0; c < NUM_CLIENTS; c++) begin : gen_decomp_per_client
            logic decomp_ready_c;
            
            // Multiplex input to this client
            logic [511:0] comp_in_c;
            logic [63:0] comp_valid_c;
            logic comp_req;
            
            assign comp_req = (chunk_client == c[5:0]) & chunk_route_valid;
            assign comp_in_c = (comp_req) ? nvme_data : 512'b0;
            assign comp_valid_c = (comp_req) ? nvme_valid : 64'b0;
            
            decompressor #(
                .INPUT_WIDTH(512),
                .OUTPUT_WIDTH(512),
                .MAX_SYMBOLS(256),
                .CHUNK_SIZE(4194304),
                .HUFF_TABLE_SIZE(4096)
            ) decomp_inst (
                .clk(clk),
                .rst_n(rst_n),
                .comp_data_in(comp_in_c),
                .comp_data_valid(comp_valid_c[6:0]),
                .comp_data_len(nvme_len),
                .comp_block_start(1'b0),  // TODO: track block boundaries
                .comp_block_last(1'b0),
                .comp_data_ready(decomp_ready_c),
                .decomp_data_out(client_data[c]),
                .decomp_data_valid(client_valid[c]),
                .decomp_data_len(),
                .decomp_block_valid(),
                .decomp_ready(client_ready[c]),
                .huff_cfg_valid(1'b0),
                .crc32_out(),
                .crc32_valid()
            );
        end
    endgenerate
    
    // Aggregated ready signal
    assign nvme_ready = |(client_ready >> chunk_client);

endmodule
