import os
import requests
from tqdm import tqdm
from requests.exceptions import RequestException

def download_files():
    simnames = ['IllustrisTNG_DM']
    snapnums = [90, 74, 62, 52, 44, 38, 32, 28, 24]
    
    with open('error.log', 'a') as error_log:
        # 第一层循环：模拟名称
        for simname in simnames:
            # 第二层循环：编号0-999
            for num in range(254, 1000):
                # 第三层循环：快照编号
                for snapnum in snapnums:
                    # 构造URL和文件路径
                    url = f"https://users.flatironinstitute.org/~camels/Sims/{simname}/LH/LH_{num}/groups_0{snapnum}.hdf5"
                    dir_path = os.path.join(f'/data2/wbc/camels/{simname}', f"LH_{num}/groups")
                    file_name = f"groups_0{snapnum}.hdf5"
                    file_path = os.path.join(dir_path, file_name)
                    
                    # 创建目标目录
                    os.makedirs(dir_path, exist_ok=True)
                    
                    # 删除已存在的文件（确保重新下载）
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            print(f"已删除不完整文件：{file_path}")
                        except OSError as e:
                            print(f"文件删除失败：{file_path} - {str(e)}")
                            continue
                    
                    try:
                        # 发起带超时的请求
                        response = requests.get(url, stream=True, timeout=30)
                        response.raise_for_status()
                        
                        # 获取文件总大小
                        total_size = int(response.headers.get('content-length', 0))
                        
                        # 创建带进度条的写入过程
                        with open(file_path, 'wb') as f, tqdm(
                            desc=f"{simname}-{num}-{snapnum}".ljust(30),
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for chunk in response.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                bar.update(size)
                                
                        # 验证下载完整性
                        actual_size = os.path.getsize(file_path)
                        if total_size > 0 and actual_size != total_size:
                            raise IOError(f"文件大小不匹配：预期{total_size}，实际{actual_size}")
                            
                    except (RequestException, IOError) as e:
                        # 发生错误时删除不完整文件
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except OSError as cleanup_error:
                                print(f"清理失败：{file_path} - {str(cleanup_error)}")
                        
                        # 记录错误日志
                        error_entry = f"{simname} {num} {snapnum} # {str(e)}\n"
                        error_log.write(error_entry)
                        error_log.flush()
                        print(f"下载失败：{error_entry.strip()}")

if __name__ == "__main__":
    download_files()