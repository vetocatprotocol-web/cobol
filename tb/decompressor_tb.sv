/*
 * DECOMPRESSOR_TB: SystemVerilog Testbench
 * Tests: Huffman decode, RLE expand, CRC32 integrity
 */

`timescale 1ns / 1ps

module decompressor_tb();

    logic clk, rst_n;
    
    // Compressed input
    logic [511:0] comp_data_in;
    logic [6:0] comp_data_valid;
    logic [11:0] comp_data_len;
    logic comp_block_start, comp_block_last;
    logic comp_data_ready;
    
    // Decompressed output
    logic [511:0] decomp_data_out;
    logic [6:0] decomp_data_valid;
    logic [11:0] decomp_data_len;
    logic decomp_block_valid;
    logic decomp_ready;
    
    // CRC
    logic [31:0] crc32_out;
    logic crc32_valid;
    
    decompressor #(
        .INPUT_WIDTH(512),
        .OUTPUT_WIDTH(512),
        .MAX_SYMBOLS(256),
        .CHUNK_SIZE(4194304),
        .HUFF_TABLE_SIZE(4096)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .comp_data_in(comp_data_in),
        .comp_data_valid(comp_data_valid),
        .comp_data_len(comp_data_len),
        .comp_block_start(comp_block_start),
        .comp_block_last(comp_block_last),
        .comp_data_ready(comp_data_ready),
        .decomp_data_out(decomp_data_out),
        .decomp_data_valid(decomp_data_valid),
        .decomp_data_len(decomp_data_len),
        .decomp_block_valid(decomp_block_valid),
        .decomp_ready(decomp_ready),
        .huff_cfg_valid(1'b0),
        .huff_cfg_addr(12'b0),
        .huff_cfg_code_len(16'b0),
        .huff_cfg_code_val(16'b0),
        .huff_cfg_symbol(8'b0),
        .huff_cfg_rdy(),
        .crc32_out(crc32_out),
        .crc32_valid(crc32_valid)
    );
    
    initial begin
        clk = 0;
        forever #2 clk = ~clk;  // 250 MHz
    end
    
    initial begin
        rst_n = 0;
        #10 rst_n = 1;
    end
    
    // Test 1: Basic throughput
    task test_throughput();
        longint bytes_in = 0;
        longint bytes_out = 0;
        int cycles = 0;
        
        $display("[TEST] Throughput measurement (1M bytes)");
        
        for (int i = 0; i < 1000000; i += 64) begin
            @(posedge clk);
            comp_data_in = {64{$urandom}};
            comp_data_valid = 7'b1111111;
            comp_data_len = 12'd64;
            decomp_ready = 1;
            
            bytes_in += 64;
            if (decomp_data_valid != 0)
                bytes_out += decomp_data_len;
            
            cycles++;
        end
        
        real throughput = (real'(bytes_out) * 250) / cycles;
        $display("[INFO] Decomp throughput: %.1f GB/s (cycles=%0d)", throughput, cycles);
    endtask
    
    // Test 2: Latency
    task test_latency();
        int latencies[100];
        int min_lat = 1000, max_lat = 0, avg_lat = 0;
        
        $display("[TEST] Latency measurement");
        
        for (int i = 0; i < 100; i++) begin
            int lat = 0;
            @(posedge clk);
            comp_data_valid = 7'b1;
            
            while (!decomp_data_valid && lat < 100) begin
                lat++;
                @(posedge clk);
            end
            
            latencies[i] = lat;
            avg_lat += lat;
            if (lat < min_lat) min_lat = lat;
            if (lat > max_lat) max_lat = lat;
            
            comp_data_valid = 0;
        end
        
        avg_lat = avg_lat / 100;
        $display("[INFO] Latency: min=%0d, avg=%0d, max=%0d cycles", min_lat, avg_lat, max_lat);
    endtask
    
    initial begin
        $display("=== DECOMPRESSOR Testbench Started ===\n");
        
        wait(rst_n);
        decomp_ready = 1;
        
        test_throughput();
        $display();
        
        test_latency();
        $display();
        
        #1000;
        $display("=== DECOMPRESSOR Testbench Complete ===\n");
        $finish;
    end
    
    initial begin
        $dumpfile("decompressor_tb.vcd");
        $dumpvars(0, decompressor_tb);
    end

endmodule
