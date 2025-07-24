----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 03/11/2025 06:38:33 PM
-- Design Name: 
-- Module Name: sesion1 - Behavioral
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

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity sesion1 is
    Port ( SW : in STD_LOGIC_VECTOR (15 downto 0);
           LED : out STD_LOGIC_VECTOR (15 downto 0);
           BTN : in STD_LOGIC_VECTOR (4 downto 0);
           CLK : in STD_LOGIC
           );
end sesion1;


architecture Behavioral of sesion1 is

SIGNAL LED_ACTIVE : STD_LOGIC_VECTOR (15 downto 0) := "0000000000000111";
SIGNAL DIR : boolean := true; -- derecha izquierda (true) ;   izquierda derecha (false);
SIGNAL CLK_COUNT : integer range 0 to 1_000_000 := 0;
begin

process (CLK, BTN)
begin
        
        if rising_edge(CLK) then -- cuando llegue a 25k ciclos, reiniciamos la frecuencia
            -- 
           if BTN(4) = '1' then
            -- Si presionas el boton del centro
            LED_ACTIVE <= "0000000000000111";
            DIR <= true;
            CLK_COUNT <= 0;
           end if;
           
           if CLK_COUNT = 1_000_000 then
            CLK_COUNT <= 0;
            -- si llegamos a un extremo, cammbiamos la direccion
            if LED_ACTIVE(14) = '1' then
                DIR <= false;
            elsif LED_ACTIVE(1) = '1' then
                DIR <=  true;
            end if;
            -- concatenamos en funcion de la direccion
            if DIR then
                LED_ACTIVE <= LED_ACTIVE(14 downto 0) & '0';-- desplazamos el vector hacia la izquierda
            else
                LED_ACTIVE <= '0' & LED_ACTIVE(15 downto 1);-- desplazamos el vector hacia la derecha
            end if;
            
           else
            CLK_COUNT <= CLK_COUNT + 1;
           end if;
            --
            
        end if;
end process;

LED <= LED_ACTIVE;

end Behavioral;
