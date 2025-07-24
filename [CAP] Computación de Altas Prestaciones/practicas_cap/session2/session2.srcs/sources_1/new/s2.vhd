----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 03/11/2025 06:38:33 PM
-- Design Name: 
-- Module Name: s2 - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description:
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity s2 is
    Port ( CLK : in STD_LOGIC;
           SSEG_CA : out STD_LOGIC_VECTOR(6 downto 0);
           SSEG_AN : out STD_LOGIC_VECTOR(3 downto 0));
end s2;

architecture Behavioral of s2 is
    -- Frecuencia para refresco del display
    constant CLK_LIMIT_MUX : integer := 100_000;
    signal CLK_COUNT_MUX : integer range 0 to CLK_LIMIT_MUX := 0;

    -- Frecuencia para desplazamiento de letras (más lento)
    constant CLK_LIMIT_SCROLL : integer := 100_000_000;
    signal CLK_COUNT_SCROLL : integer range 0 to CLK_LIMIT_SCROLL := 0;

    signal selection : integer range 0 to 3 := 0; -- display activo (0-3)
    signal scroll_index : integer range 0 to 6 := 0; -- índice de desplazamiento del mensaje

    signal letra : STD_LOGIC_VECTOR(6 downto 0);

    -- Letras del mensaje
    type letras_array is array (0 to 6) of STD_LOGIC_VECTOR(6 downto 0);
    signal letras : letras_array := (
        "0001000",  -- A
        "1000111",  -- L
        "0000011",  -- b
        "0000110",  -- E
        "0101111",  -- r
        "0000111",  -- t
        "1000000"   -- O
    );
begin

    -- Proceso para manejar el scroll del mensaje (desplazamiento de letras)
    process (CLK)
    begin
        if rising_edge(CLK) then
            if CLK_COUNT_SCROLL = CLK_LIMIT_SCROLL then
                CLK_COUNT_SCROLL <= 0;
                scroll_index <= (scroll_index + 1) mod 7; -- avanza el scroll
            else
                CLK_COUNT_SCROLL <= CLK_COUNT_SCROLL + 1;
            end if;
        end if;
    end process;

    -- Proceso para manejar la multiplexación de los displays
    process (CLK)
    begin
        if rising_edge(CLK) then
            if CLK_COUNT_MUX = CLK_LIMIT_MUX then
                CLK_COUNT_MUX <= 0;
                selection <= (selection + 1) mod 4; -- cambia de display activo
            else
                CLK_COUNT_MUX <= CLK_COUNT_MUX + 1;
            end if;
        end if;
    end process;

    -- Selección de letra desplazada según scroll_index
    process (selection, scroll_index)
        variable index : integer;
    begin
        -- calculamos la letra a mostrar en cada display
        index := (scroll_index + selection) mod 7;
        letra <= letras(index);
    end process;

    -- Activación del ánodo correspondiente
    SSEG_AN <= "0111" when selection = 0 else
               "1011" when selection = 1 else
               "1101" when selection = 2 else
               "1110";

    -- Salida de segmentos
    SSEG_CA <= letra;

end Behavioral;


