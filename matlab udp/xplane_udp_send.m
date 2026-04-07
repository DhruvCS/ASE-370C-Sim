function xplane_udp_send(throttle, elevator, aileron, rudder)
% XPLANE_UDP_SEND  Send control surface commands to X-Plane via UDP DATA packet
%
% Usage (call from a Simulink MATLAB Function block):
%   xplane_udp_send(throttle, elevator, aileron, rudder)
%
% Inputs (all scalars, normalized -1 to 1 except throttle 0 to 1):
%   throttle  : 0.0 (idle) to 1.0 (full)
%   elevator  : -1.0 (full nose down) to 1.0 (full nose up)
%   aileron   : -1.0 (full left) to 1.0 (full right)
%   rudder    : -1.0 (full left) to 1.0 (full right)
%
% X-Plane DATA packet format (little-endian):
%   Bytes 0-3   : ASCII 'DATA'
%   Bytes 4-7   : padding (0x00 0x00 0x00 0x00)
%   Then for each data group (36 bytes):
%     Bytes 0-3   : index (int32, little-endian)
%     Bytes 4-35  : 8 floats (single precision, little-endian)
%
% X-Plane data indices used:
%   Index 8  : flight controls  [elevator, aileron, rudder, ...]
%   Index 25 : throttle         [throttle, ...]

    XPLANE_IP   = '127.0.0.1';
    XPLANE_PORT = 49001;

    % --- Build flight controls packet (index 8) ---
    % Slot layout for index 8: [elevator, aileron, rudder, nosewheel, ...]
    ctrl_floats = single([elevator, aileron, rudder, 0, 0, 0, 0, 0]);
    ctrl_packet = build_data_packet(int32(8), ctrl_floats);

    % --- Build throttle packet (index 25) ---
    % Slot layout for index 25: [throttle_eng1, throttle_eng2, ...]
    thr_floats = single([throttle, throttle, 0, 0, 0, 0, 0, 0]);
    thr_packet = build_data_packet(int32(25), thr_floats);

    % --- Send both packets ---
    u = udpport("byte", "LocalPort", 0);  % ephemeral local port
    write(u, ctrl_packet, "uint8", XPLANE_IP, XPLANE_PORT);
    write(u, thr_packet,  "uint8", XPLANE_IP, XPLANE_PORT);
    clear u;  % close socket immediately
end


function packet = build_data_packet(index, floats_single)
% Builds a complete X-Plane DATA UDP packet for one data group.
%
% Args:
%   index         : int32 scalar, the DATA index number
%   floats_single : 1x8 single array of values
%
% Returns:
%   packet : uint8 row vector ready to send

    % Header: 'DATA' + null padding byte
    header = uint8([68, 65, 84, 65, 0]);  % 'D','A','T','A', 0x00

    % Index as 4 bytes little-endian int32
    idx_bytes = typecast(int32(index), 'uint8');

    % 8 floats as 32 bytes little-endian
    float_bytes = typecast(floats_single, 'uint8');

    packet = [header, idx_bytes, float_bytes];
end
