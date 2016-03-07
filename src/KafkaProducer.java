import java.util.Properties;

import kafka.javaapi.producer.Producer;
import kafka.producer.KeyedMessage;
import kafka.producer.ProducerConfig;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

public class KafkaProducer {
	private final Producer<String, String> producer;
	public final static String TOPIC = "TEST-TOPIC";

	private KafkaProducer() {
		Properties props = new Properties();
		props.put("metadata.broker.list", "master:9092");
		props.put("seriallizer.class", "kafka.serializer.StringEncoder");
		props.put("key.seriallizer.class", "kafka.serializer.StringEncoder");
		props.put("request.required.acks", "-1");
		producer = new Producer<String, String>(new ProducerConfig(props));
	}

	void read(String file) {
		String s = null;
		StringBuffer sb = new StringBuffer();
		File f = new File(file);
		if (f.exists()) {
			try {
				@SuppressWarnings("resource")
				BufferedReader br = new BufferedReader(new InputStreamReader(
						new FileInputStream(f)));
				while ((s = br.readLine()) != null) {
					sb.append(s);
					String key = String.valueOf(s);
					String data = "hello kafka message " + key;
					producer.send(new KeyedMessage<String, String>("test",
							null, key));
					System.out.println(data);
				}
			} catch (Exception e) {
				// TODO: handle exception
			}
		} else {
			System.out.println("file not exsist！");
		}
	}

	public static void main(String[] args) {
		new KafkaProducer().read("/home/zkpk/data/k.txt");
	}

}
