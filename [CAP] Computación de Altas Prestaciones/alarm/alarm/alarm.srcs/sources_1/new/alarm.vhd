-- Alarm Clock with Hour Edit and Stopwatch Modes
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity alarm is
    Port (
        CLK      : in STD_LOGIC;
        SSEG_CA  : out STD_LOGIC_VECTOR(6 downto 0);
        SSEG_AN  : out STD_LOGIC_VECTOR(3 downto 0);
        LED      : out STD_LOGIC_VECTOR(15 downto 0);
        BTN      : in STD_LOGIC_VECTOR(4 downto 0);
        SW       : in STD_LOGIC_VECTOR(15 downto 0)
    );
end alarm;

architecture Behavioral of alarm is
    constant SEC_LIMIT : integer := 100_000_000;
    constant CLK_LIMIT : integer := 100_000;

    signal sec_count, clk_count : integer := 0;
    signal seconds, minutes : integer range 0 to 59 := 0;
    signal hours : integer range 0 to 23 := 0;

    signal letra : STD_LOGIC_VECTOR(6 downto 0);
    signal digits : STD_LOGIC_VECTOR(3 downto 0);
    signal selection : integer range 0 to 3 := 0;

    signal led_out : STD_LOGIC_VECTOR(15 downto 0) := (others => '0');

    -- Edit Mode
    signal edit_mode, chrono_mode, alarm_edit_mode : STD_LOGIC := '0';
    signal edit_position : integer range 0 to 3 := 0;
    signal blink_count : integer := 0;
    signal blink : STD_LOGIC := '1';

    -- Stopwatch
    signal chrono_seconds, chrono_minutes : integer range 0 to 59 := 0;
    signal chrono_sec_count : integer := 0;
    signal saved_minutes, saved_seconds : integer range 0 to 59 := 0;

    -- Alarm
    signal alarm_hours : integer range 0 to 23 := 0;
    signal alarm_minutes : integer range 0 to 59 := 0;
    signal alarm_active : STD_LOGIC := '0';
    signal alarm_activated : STD_LOGIC := '0';
    signal alarm_led_counter : integer range 0 to 30 := 0;

    -- Buttons
    signal btn_prev, btn_edge : STD_LOGIC_VECTOR(4 downto 0) := (others => '0');
begin

    -- Blink
    process(CLK)
    begin
        if rising_edge(CLK) then
            if blink_count = 49_999_999 then
                blink_count <= 0;
                blink <= not blink;
            else
                blink_count <= blink_count + 1;
            end if;
        end if;
    end process;

    -- LED blink (seconds)
    process(seconds)
    begin
        if seconds mod 2 = 0 then
            led_out(0) <= '1';
        else
            led_out(0) <= '0';
        end if;
    end process;

    -- Main Clock Process
    process(CLK)
    begin
        if rising_edge(CLK) then
            btn_edge <= BTN and not btn_prev;
            btn_prev <= BTN;

            -- Modo edición de alarma
            if SW(0) = '1' and SW(1) = '1' then
                alarm_edit_mode <= '1';
                edit_mode <= '0';
                chrono_mode <= '0';
            end if;

            -- Alarma programada
            if alarm_edit_mode = '1' and SW(0) = '0' and SW(1) = '0' then
                alarm_edit_mode <= '0';
                alarm_active <= '1';
            end if;

            -- Desactivar alarma
            if alarm_active = '1' and alarm_activated = '1' and btn_edge(4) = '1' then
                alarm_activated <= '0';
                alarm_led_counter <= 0;
            end if;

            -- BTN(4) acciones
            if btn_edge(4) = '1' then
                if SW(0) = '1' and SW(1) = '0' then
                    chrono_mode <= not chrono_mode;
                    if chrono_mode = '1' then
                        chrono_minutes <= 0;
                        chrono_seconds <= 0;
                        chrono_sec_count <= 0;
                        saved_minutes <= minutes;
                        saved_seconds <= seconds;
                    end if;
                elsif SW(0) = '1' and SW(1) = '1' then
                    alarm_active <= '0';
                    alarm_activated <= '0';
                    alarm_led_counter <= 0;
                else
                    edit_mode <= not edit_mode;
                end if;
            end if;

            -- Edición
            if edit_mode = '1' or alarm_edit_mode = '1' then
                led_out(1) <= '1';

                if btn_edge(1) = '1' then
                    edit_position <= (edit_position - 1 + 4) mod 4;
                elsif btn_edge(2) = '1' then
                    edit_position <= (edit_position + 1) mod 4;
                end if;

                if btn_edge(0) = '1' then
                    case edit_position is
                        when 0 => if alarm_edit_mode = '1' then alarm_hours <= (alarm_hours + 10) mod 24; else hours <= (hours + 10) mod 24; end if;
                        when 1 => if alarm_edit_mode = '1' then alarm_hours <= (alarm_hours + 1) mod 24; else hours <= (hours + 1) mod 24; end if;
                        when 2 => if alarm_edit_mode = '1' then alarm_minutes <= (alarm_minutes + 10) mod 60; else minutes <= (minutes + 10) mod 60; end if;
                        when 3 => if alarm_edit_mode = '1' then alarm_minutes <= (alarm_minutes + 1) mod 60; else minutes <= (minutes + 1) mod 60; end if;
                        when others => null;
                    end case;
                elsif btn_edge(3) = '1' then
                    case edit_position is
                        when 0 => if alarm_edit_mode = '1' then alarm_hours <= (alarm_hours - 10 + 24) mod 24; else hours <= (hours - 10 + 24) mod 24; end if;
                        when 1 => if alarm_edit_mode = '1' then alarm_hours <= (alarm_hours - 1 + 24) mod 24; else hours <= (hours - 1 + 24) mod 24; end if;
                        when 2 => if alarm_edit_mode = '1' then alarm_minutes <= (alarm_minutes - 10 + 60) mod 60; else minutes <= (minutes - 10 + 60) mod 60; end if;
                        when 3 => if alarm_edit_mode = '1' then alarm_minutes <= (alarm_minutes - 1 + 60) mod 60; else minutes <= (minutes - 1 + 60) mod 60; end if;
                        when others => null;
                    end case;
                end if;
            else
                led_out(1) <= '0';

                if sec_count = SEC_LIMIT then
                    sec_count <= 0;

                    -- cronómetro
                    if chrono_mode = '1' then
                        if chrono_seconds = 59 then
                            chrono_seconds <= 0;
                            chrono_minutes <= (chrono_minutes + 1) mod 60;
                        else
                            chrono_seconds <= chrono_seconds + 1;
                        end if;
                    end if;

                    -- reloj
                    if seconds = 59 then
                        seconds <= 0;
                        if minutes = 59 then
                            minutes <= 0;
                            hours <= (hours + 1) mod 24;
                        else
                            minutes <= minutes + 1;
                        end if;
                    else
                        seconds <= seconds + 1;
                    end if;

                    -- Contador de alarma
                    if alarm_activated = '1' and alarm_led_counter < 30 then
                        alarm_led_counter <= alarm_led_counter + 1;
                    end if;
                else
                    sec_count <= sec_count + 1;
                end if;
            end if;

            -- Activar alarma
            if alarm_active = '1' and alarm_activated = '0' then
                if hours = alarm_hours and minutes = alarm_minutes then
                    alarm_activated <= '1';
                    alarm_led_counter <= 0;
                end if;
            end if;

            -- LED alarma progresiva
            if alarm_led_counter > 0  then led_out(15) <= '1'; else led_out(15) <= '0'; end if;
            if alarm_led_counter >= 5  then led_out(14) <= '1'; else led_out(14) <= '0'; end if;
            if alarm_led_counter >= 10 then led_out(13) <= '1'; else led_out(13) <= '0'; end if;
            if alarm_led_counter >= 15 then led_out(12) <= '1'; else led_out(12) <= '0'; end if;
            if alarm_led_counter >= 20 then led_out(11) <= '1'; else led_out(11) <= '0'; end if;
            if alarm_led_counter >= 25 then led_out(10) <= '1'; else led_out(10) <= '0'; end if;
        end if;
    end process;

    -- Multiplexado display
    process(CLK)
    begin
        if rising_edge(CLK) then
            if clk_count = CLK_LIMIT then
                clk_count <= 0;
                selection <= (selection + 1) mod 4;
            else
                clk_count <= clk_count + 1;
            end if;
        end if;
    end process;

    -- Lógica visualización
    process(selection, minutes, hours, chrono_mode, chrono_minutes, chrono_seconds,
            edit_mode, alarm_edit_mode, edit_position, blink)
        variable temp_digit : integer := 0;
        variable visible : boolean := true;
    begin
        visible := not ((edit_mode = '1' or alarm_edit_mode = '1') and selection = edit_position and blink = '0');

        if chrono_mode = '1' then
            case selection is
                when 0 => temp_digit := chrono_minutes / 10;
                when 1 => temp_digit := chrono_minutes mod 10;
                when 2 => temp_digit := chrono_seconds / 10;
                when 3 => temp_digit := chrono_seconds mod 10;
                when others => temp_digit := 0;
            end case;
        elsif alarm_edit_mode = '1' then
            case selection is
                when 0 => temp_digit := alarm_hours / 10;
                when 1 => temp_digit := alarm_hours mod 10;
                when 2 => temp_digit := alarm_minutes / 10;
                when 3 => temp_digit := alarm_minutes mod 10;
                when others => temp_digit := 0;
            end case;
        else
            case selection is
                when 0 => temp_digit := hours / 10;
                when 1 => temp_digit := hours mod 10;
                when 2 => temp_digit := minutes / 10;
                when 3 => temp_digit := minutes mod 10;
                when others => temp_digit := 0;
            end case;
        end if;

        if visible then
            digits <= std_logic_vector(to_unsigned(temp_digit, 4));
        else
            digits <= "1111";
        end if;
    end process;

    -- BCD a 7 segmentos
    process(digits)
    begin
        case digits is
            when "0000" => letra <= "1000000"; -- 0
            when "0001" => letra <= "1111001"; -- 1
            when "0010" => letra <= "0100100"; -- 2
            when "0011" => letra <= "0110000"; -- 3
            when "0100" => letra <= "0011001"; -- 4
            when "0101" => letra <= "0010010"; -- 5
            when "0110" => letra <= "0000010"; -- 6
            when "0111" => letra <= "1111000"; -- 7
            when "1000" => letra <= "0000000"; -- 8
            when "1001" => letra <= "0010000"; -- 9
            when others => letra <= "1111111"; -- apagado
        end case;
    end process;

    SSEG_AN <= "0111" when selection = 0 else
               "1011" when selection = 1 else
               "1101" when selection = 2 else
               "1110";
    SSEG_CA <= letra;
    LED <= led_out;
end Behavioral;
