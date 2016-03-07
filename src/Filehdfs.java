import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class Filehdfs {
	public String getYHDSXCategoryIdStr(String filePath) {
		final String DELIMITER = new String(new byte[] { 1 });
		final String INNER_DELIMITER = ",";

		// 遍历目录下的所有文件
		BufferedReader br = null;
		String yhdsxCategoryIdStr = null;
		try {
			FileSystem fs = FileSystem.get(new Configuration());
			FileStatus[] status = fs.listStatus(new Path(filePath));
			for (FileStatus file : status) {
				if (!file.getPath().getName().startsWith("part-")) {
					continue;
				}

				FSDataInputStream inputStream = fs.open(file.getPath());
				br = new BufferedReader(new InputStreamReader(inputStream));

				String line = null;
				while (null != (line = br.readLine())) {
					String[] strs = line.split(DELIMITER);
					String categoryId = strs[0];
					String categorySearchName = strs[9];
					if (-1 != categorySearchName.indexOf("0-956955")) {
						yhdsxCategoryIdStr += (categoryId + INNER_DELIMITER);
					}
				}// end of while
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return yhdsxCategoryIdStr;
	}
}
