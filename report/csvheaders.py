# Generate the CSV headers and save to a file for download
zones = 64
bins_per_zone = 18  # Bins [0..17]

headers = ["SensorMode", "MessageType"]
for z in range(zones):
    headers.append(f"Z{z}_Status")
    headers.append(f"Z{z}_Distance")
    headers.append(f"Z{z}_Ambient")
    for b in range(bins_per_zone):
        headers.append(f"Z{z}_Bin{b}")

print(",".join(headers))
# # Save a single-row CSV with only headers
# import csv, os
# output_path = "/mnt/data/vl53lmz_headers.csv"
# with open(output_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(headers)
#
# # Also return the header line as a string (preview)
# header_line = ",".join(headers)
# header_line[:5000]  # preview a truncated portion to avoid overwhelming the chat
