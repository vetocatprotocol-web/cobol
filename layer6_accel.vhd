-- layer6_accel.vhd (pseudocode VHDL)
-- Acceleration module for Layer 6 pattern matching

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity layer6_accel is
    port (
        clk        : in  std_logic;
        rst        : in  std_logic;
        data_in    : in  std_logic_vector(511 downto 0);
        data_out   : out std_logic_vector(511 downto 0);
        valid_out  : out std_logic
    );
end entity;

architecture rtl of layer6_accel is
    type bram_t is array(0 to 4095) of std_logic_vector(15 downto 0);
    signal pattern_mem : bram_t;
    signal matched_id  : unsigned(31 downto 0);
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                matched_id <= (others => '0');
                valid_out  <= '0';
            else
                -- sliding window compare pseudo-loop
                for i in 0 to 31 loop
                    if data_in(i*16+15 downto i*16) = pattern_mem(i) then
                        matched_id <= to_unsigned(i, 32);
                    end if;
                end loop;
                data_out <= data_in xor std_logic_vector(matched_id & matched_id);
                valid_out <= '1';
            end if;
        end if;
    end process;
end architecture;
