function [airspeed, heading, pitch, roll, lat, lon, alt] = xplane_udp_receive()
% XPLANE_UDP_RECEIVE  Read and unpack X-Plane UDP broadcast (port 49000)
%
% Usage (call from a Simulink MATLAB Function block):
%   [airspeed, heading, pitch, roll, lat, lon, alt] = xplane_udp_receive()
%
% Returns:
%   airspeed  : indicated airspeed [m/s]  (X-Plane broadcasts in ktas, converted)
%   heading   : true heading [deg], 0-360
%   pitch     : pitch angle [deg], positive nose up
%   roll      : roll/bank angle [deg], positive right wing down
%   lat       : latitude [deg]
%   lon       : longitude [deg]
%   alt       : altitude MSL [m]
%
% X-Plane must have "Send network data output" enabled in Settings > Data Output
% for indices 3 (speeds), 17 (pitch/roll/heading), 20 (lat/lon/alt).
%
% DATA packet format received from X-Plane:
%   Bytes 0-4   : 'DATA\0' header
%   Then N groups of 36 bytes:
%     Bytes 0-3   : index (int32 LE)
%     Bytes 4-35  : 8 x float32 LE values
%
% Index map (slot numbers are 0-based):
%   Index 3  : speeds     -> slot 0 = Vind (ktas)
%   Index 17 : orientation-> slot 0 = pitch, slot 1 = roll, slot 2 = true hdg
%   Index 20 : position   -> slot 0 = lat, slot 1 = lon, slot 2 = alt (ft MSL)

    LISTEN_PORT  = 49000;
    TIMEOUT_S    = 0.05;   % 50 ms — match your 20 Hz Simulink rate
    KTAS_TO_MS   = 0.514444;
    FT_TO_M      = 0.3048;

    % Defaults (returned if packet not received in time)
    airspeed = 0; heading = 0; pitch = 0; roll = 0;
    lat = 0; lon = 0; alt = 0;

    u = udpport("byte", "LocalPort", LISTEN_PORT, "EnablePortSharing", true);
    u.Timeout = TIMEOUT_S;

    try
        % Read up to 4096 bytes — one X-Plane broadcast frame
        raw = read(u, 4096, "uint8");
    catch
        clear u;
        return;  % timeout — return defaults
    end
    clear u;

    if numel(raw) < 5
        return;
    end

    % Verify header 'DATA\0'
    if ~isequal(raw(1:5), uint8([68,65,84,65,0]))
        return;
    end

    % Parse groups starting at byte 6 (1-indexed)
    offset = 6;
    while offset + 35 <= numel(raw)
        idx    = double(typecast(uint8(raw(offset:offset+3)),   'int32'));
        vals   = double(typecast(uint8(raw(offset+4:offset+35)),'single'));  % 1x8

        switch idx
            case 3   % Speeds
                airspeed = vals(1) * KTAS_TO_MS;  % ktas -> m/s

            case 17  % Pitch / Roll / Heading
                pitch   = vals(1);
                roll    = vals(2);
                heading = vals(3);  % true heading 0-360

            case 20  % Lat / Lon / Alt
                lat = vals(1);
                lon = vals(2);
                alt = vals(3) * FT_TO_M;  % ft -> m
        end

        offset = offset + 36;
    end
end
