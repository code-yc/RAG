package cn.yc.dev.test;


import com.alibaba.fastjson.JSON;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.PgVectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@RunWith(SpringRunner.class)
@SpringBootTest
public class RAGTest {

    @Resource
    private OllamaChatClient ollamaChatClient;
    @Resource
    private TokenTextSplitter tokenTextSplitter;
    @Resource
    private SimpleVectorStore simpleVectorStore;
    @Resource
    private PgVectorStore pgVectorStore;

    @Test
    public void upload() {
        TikaDocumentReader reader = new TikaDocumentReader("data/file.txt");

        List<Document> documents = reader.get();
        log.info("文档读取完成，原始文档数量: {}", documents.size());

        log.info("开始文档切分...");
        List<Document> documentSplitterList = tokenTextSplitter.apply(documents);
        log.info("文档切分完成，切分后文档数量: {}", documentSplitterList.size());

        log.info("为文档添加元数据标记...");
        documents.forEach(doc -> doc.getMetadata().put("knowledge", "小明知识库"));
        documentSplitterList.forEach(doc -> doc.getMetadata().put("knowledge", "小明知识库"));
        log.info("元数据添加完成");

        log.info("开始向PostgreSQL向量数据库写入文档...");
        pgVectorStore.accept(documentSplitterList);

        log.info("文档处理完成，共处理文档{}个，切分后文档{}个", documents.size(), documentSplitterList.size());
    }


    @Test
    public void chat() {
        String message = "小明是谁？哪年出生";
        log.info("用户提问: {}", message);

        String SYSTEM_PROMPT = """
                Use the information from the DOCUMENTS section to provide accurate answers but act as if you knew this information innately.
                If unsure, simply state that you don't know.
                Another thing you need to note is that your reply must be in Chinese!
                DOCUMENTS:
                    {documents}
                """;
        log.info("系统提示模板已设置");

        log.info("开始向量数据库相似度搜索，查询: {}", message);
        SearchRequest request = SearchRequest.query(message).withTopK(5).withFilterExpression("knowledge == '小明知识库'");
        log.info("搜索参数: topK={}, 过滤条件={}", 5, "knowledge == '小明知识库'");

        List<Document> documents = pgVectorStore.similaritySearch(request);
        log.info("搜索完成，找到相关文档数量: {}", documents.size());

        for (int i = 0; i < documents.size(); i++) {
            log.info("文档[{}]内容片段: {}", i+1, documents.get(i).getContent().substring(0, Math.min(50, documents.get(i).getContent().length())) + "...");
        }

        String documentsCollectors = documents.stream().map(Document::getContent).collect(Collectors.joining());
        log.info("合并文档总长度: {} 字符", documentsCollectors.length());

        log.info("创建RAG系统消息...");
        Message ragMessage = new SystemPromptTemplate(SYSTEM_PROMPT).createMessage(Map.of("documents", documentsCollectors));

        ArrayList<Message> messages = new ArrayList<>();
        messages.add(new UserMessage(message));
        messages.add(ragMessage);
        log.info("消息准备完成，共 {} 条消息", messages.size());

        log.info("开始调用Ollama模型 deepseek-r1:1.5b...");
        ChatResponse chatResponse = ollamaChatClient.call(new Prompt(messages, OllamaOptions.create().withModel("deepseek-r1:1.5b")));
        log.info("模型调用完成");

        log.info("AI回答: {}", chatResponse.getResult().getOutput().getContent());
        log.info("完整响应: {}", JSON.toJSONString(chatResponse));
    }
}
