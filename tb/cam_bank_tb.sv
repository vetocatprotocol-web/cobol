/*
 * CAM_BANK_TB: SystemVerilog Testbench for CAM module
 * Tests: parallel probe correctness, latency, hit/miss behavior
 */

`timescale 1ns / 1ps

module cam_bank_tb();

    // ======== Clock & Reset ========
    logic clk;
    logic rst_n;
    
    // ======== Test Signals ========
    logic [511:0] data_in;
    logic [3:0] data_valid;
    logic [7:0] data_len;
    
    logic [95:0] probe_key;
    logic probe_valid;
    logic probe_last;
    logic probe_ready;
    
    logic [31:0] match_id;
    logic match_valid;
    logic match_hit;
    logic [7:0] match_len;
    
    // ======== Instance ========
    cam_bank #(
        .DATA_WIDTH(512),
        .KEY_WIDTH(96),
        .CAM_DEPTH(65536),
        .NUM_PROBES(32),
        .HBM_DEPTH(1048576),
        .PIPELINE_DEPTH(5)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_valid(data_valid),
        .data_len(data_len),
        .probe_key(probe_key),
        .probe_valid(probe_valid),
        .probe_last(probe_last),
        .probe_ready(probe_ready),
        .match_id(match_id),
        .match_valid(match_valid),
        .match_hit(match_hit),
        .match_len(match_len),
        .hbm_rd_valid(),
        .hbm_rd_addr(),
        .hbm_rd_data(128'b0),
        .hbm_rd_rdy(1'b1),
        .hbm_wr_valid(),
        .hbm_wr_addr(),
        .hbm_wr_data(),
        .hbm_wr_rdy(1'b1),
        .nvme_rd_valid(),
        .nvme_rd_addr(),
        .nvme_rd_data(512'b0),
        .nvme_rd_rdy(1'b1),
        .cfg_valid(1'b0),
        .cfg_addr(32'b0),
        .cfg_key(96'b0),
        .cfg_match_id(32'b0),
        .cfg_len(8'b0),
        .cfg_is_hbm(1'b0),
        .cfg_rdy()
    );
    
    // ======== Clock generation ========
    initial begin
        clk = 0;
        forever #2 clk = ~clk;  // 250 MHz
    end
    
    // ======== Reset ========
    initial begin
        rst_n = 0;
        #10 rst_n = 1;
    end
    
    // ======== Test procedures ========
    
    // Test 1: Basic probe on miss
    task test_basic_miss();
        @(posedge clk);
        probe_key = 96'hDEADBEEF_11223344_55667788;
        probe_valid = 1;
        
        repeat(10) @(posedge clk);
        probe_valid = 0;
        
        if (match_hit == 0) begin
            $display("[PASS] Basic miss test");
        end else begin
            $display("[FAIL] Expected miss, got hit");
        end
    endtask
    
    // Test 2: Parallel probe throughput
    task test_throughput();
        int probes_sent = 0;
        int probes_recv = 0;
        
        for (int i = 0; i < 1000; i++) begin
            @(posedge clk);
            probe_key = 96'h_AABBCCDD11223344 + i;
            probe_valid = 1;
            probes_sent++;
            
            if (match_valid)
                probes_recv++;
        end
        
        probe_valid = 0;
        repeat(10) @(posedge clk);
        
        $display("[INFO] Throughput: sent=%0d, received=%0d", probes_sent, probes_recv);
    endtask
    
    // Test 3: Latency measurement
    task test_latency();
        int latency;
        int latencies[100];
        int min_lat = 1000;
        int max_lat = 0;
        
        for (int i = 0; i < 100; i++) begin
            latency = 0;
            @(posedge clk);
            probe_valid = 1;
            
            while(!match_valid && latency < 100) begin
                latency++;
                @(posedge clk);
            end
            
            latencies[i] = latency;
            if (latency < min_lat) min_lat = latency;
            if (latency > max_lat) max_lat = latency;
            
            probe_valid = 0;
            @(posedge clk);
        end
        
        $display("[INFO] Latency: min=%0d cycles, max=%0d cycles", min_lat, max_lat);
    endtask
    
    // Main test sequence
    initial begin
        $display("=== CAM_BANK Testbench Started ===\n");
        
        wait(rst_n);
        $display("[INFO] Reset complete\n");
        
        test_basic_miss();
        $display();
        
        test_throughput();
        $display();
        
        test_latency();
        $display();
        
        #100;
        $display("=== CAM_BANK Testbench Complete ===\n");
        $finish;
    end
    
    // Waveform dump
    initial begin
        $dumpfile("cam_bank_tb.vcd");
        $dumpvars(0, cam_bank_tb);
    end

endmodule
